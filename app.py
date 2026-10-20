"""Flask 后端 + pywebview 原生窗口 + 实时推流"""
from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import sys
import threading
import time
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 平台自适应工作目录 =====
HOME = os.path.expanduser("~")
if platform.system() == "Darwin":
    WORK_DIR = Path(HOME) / "Library" / "Application Support" / "车辆检测系统"
else:
    WORK_DIR = Path(HOME) / "车辆检测系统数据"
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
print(f"工作目录: {WORK_DIR}")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative)


app = Flask(__name__, template_folder=resource_path("templates"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_DIR = WORK_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = WORK_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR = WORK_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

state = {
    "video_path": None, "config": {}, "processing": False,
    "progress": 0, "result": None, "track_df": None,
    "summary": None, "video_result": None, "charts": [], "error": None,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"ok": False, "msg": "没有文件"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"ok": False, "msg": "文件名为空"}), 400
    from werkzeug.utils import secure_filename
    filename = secure_filename(f.filename) or "video.mp4"
    save_path = UPLOAD_DIR / filename
    f.save(str(save_path))
    import cv2
    cap = cv2.VideoCapture(str(save_path))
    ret, first = cap.read()
    cap.release()
    if not ret:
        return jsonify({"ok": False, "msg": "无法读取视频"}), 400
    cv2.imwrite(str(UPLOAD_DIR / "preview.jpg"), first)
    state["video_path"] = str(save_path)
    return jsonify({"ok": True, "filename": filename})


@app.route("/preview")
def preview():
    return send_file(str(UPLOAD_DIR / "preview.jpg"), mimetype="image/jpeg")


# ==================== 【核心】MJPEG 实时推流 ====================
@app.route("/stream")
def stream():
    """MJPEG 流：持续把当前处理帧推送给浏览器"""
    from detector import get_current_frame

    def generate():
        while True:
            frame = get_current_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # 约 33 fps

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def compute_statistics(passed_df, track_df, config):
    pixel_per_meter = config.get("pixel_per_meter", 30.0)
    speed_calib = config.get("speed_calib_coeff", 1.5)

    if passed_df.empty or track_df.empty:
        return {}, pd.DataFrame()

    vehicle_traj = {}
    for tid, group in track_df.groupby("车辆ID"):
        group = group.sort_values("帧")
        vehicle_traj[int(tid)] = list(zip(group["时间"], group["中心点Y"]))

    def fit_speed(points):
        if len(points) < 3:
            return np.nan
        times = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        A = np.vstack([times, np.ones(len(times))]).T
        try:
            a, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        except Exception:
            return np.nan
        speed_mps = a / pixel_per_meter * speed_calib
        if abs(speed_mps) > 50:
            return np.nan
        return abs(speed_mps) * 3.6

    speeds = [fit_speed(vehicle_traj.get(int(r["车辆ID"]), []))
              for _, r in passed_df.iterrows()]

    passed_df = passed_df.copy()
    passed_df["平均速度_kmh"] = speeds
    passed_df["车长_米"] = passed_df["宽度"] / pixel_per_meter

    def limit(t):
        return config.get("high_speed_thresh", 100.0) if "小型车" in t else config.get("low_speed_thresh", 20.0)

    passed_df["限速_kmh"] = passed_df["车型"].apply(limit)
    passed_df["是否超速"] = passed_df["平均速度_kmh"] > passed_df["限速_kmh"]
    passed_df["超速数值"] = (passed_df["平均速度_kmh"] - passed_df["限速_kmh"]).where(
        passed_df["是否超速"], 0.0)

    total_veh = len(passed_df)
    class_counts = passed_df["车型"].value_counts().to_dict()
    avg_speed = passed_df["平均速度_kmh"].mean() if not passed_df["平均速度_kmh"].isna().all() else 0.0
    avg_length = passed_df["车长_米"].mean()
    total_time_s = max(track_df["时间"].max() - track_df["时间"].min(), 1)

    lane_metrics = {}
    for lane_id in sorted(passed_df["车道编号"].unique()):
        sub = passed_df[passed_df["车道编号"] == lane_id]
        n = len(sub)
        flow = n / total_time_s * 3600
        density = n / 50.0
        occ = sub["车长_米"].sum() / (50.0 * total_time_s)
        sp = sub["平均速度_kmh"].mean() if not sub["平均速度_kmh"].isna().all() else 0.0
        sub_sorted = sub.sort_values("时间")
        ys = sub_sorted["中心点Y"].values
        ts = sub_sorted["时间"].values
        hw_m, hw_t = [], []
        for i in range(1, len(ys)):
            hw_m.append(abs(ys[i] - ys[i - 1]) / pixel_per_meter)
            hw_t.append(ts[i] - ts[i - 1])
        lane_metrics[int(lane_id)] = {
            "流量_辆每小时": round(flow, 1),
            "密度_辆每米": round(density, 4),
            "时间占有率": round(occ, 4),
            "平均速度_kmh": round(sp, 2),
            "平均车头间距_m": round(np.mean(hw_m), 2) if hw_m else 0.0,
            "平均车头时距_s": round(np.mean(hw_t), 2) if hw_t else 0.0,
        }

    summary = {
        "总车辆数": total_veh,
        "车型统计": class_counts,
        "平均车速_kmh": round(avg_speed, 2),
        "平均车长_m": round(avg_length, 2),
        "总时长_s": round(total_time_s, 1),
        "分车道": lane_metrics,
    }
    return summary, passed_df


def generate_charts(passed_df, track_df, summary, config):
    chart_files = []
    if passed_df.empty:
        return chart_files

    pixel_per_meter = config.get("pixel_per_meter", 30.0)
    speed_calib = config.get("speed_calib_coeff", 1.5)
    lane_ids = sorted(summary.get("分车道", {}).keys())

    for f in CHART_DIR.glob("*.png"):
        try:
            f.unlink()
        except Exception:
            pass

    if lane_ids:
        # 1. 平均速度
        vals = [summary["分车道"][l]["平均速度_kmh"] for l in lane_ids]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar([f"车道{l}" for l in lane_ids], vals, color="#1f77b4")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("各车道平均速度", fontsize=13, fontweight="bold")
        ax.set_ylabel("速度 (km/h)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout(); plt.savefig(CHART_DIR / "speed.png", dpi=120); plt.close()
        chart_files.append("speed.png")

        # 2. 密度
        vals = [summary["分车道"][l]["密度_辆每米"] for l in lane_ids]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar([f"车道{l}" for l in lane_ids], vals, color="#942654")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("各车道交通密度", fontsize=13, fontweight="bold")
        ax.set_ylabel("密度 (辆/m)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout(); plt.savefig(CHART_DIR / "density.png", dpi=120); plt.close()
        chart_files.append("density.png")

        # 3. 流量
        vals = [summary["分车道"][l]["流量_辆每小时"] for l in lane_ids]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar([f"车道{l}" for l in lane_ids], vals, color="#ff8800")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("车道流量", fontsize=13, fontweight="bold")
        ax.set_ylabel("流量 (辆/h)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout(); plt.savefig(CHART_DIR / "flow.png", dpi=120); plt.close()
        chart_files.append("flow.png")

        # 4. 占有率
        vals = [summary["分车道"][l]["时间占有率"] * 100 for l in lane_ids]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar([f"车道{l}" for l in lane_ids], vals, color="#c83c23")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("车道时间占有率", fontsize=13, fontweight="bold")
        ax.set_ylabel("占有率 (%)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout(); plt.savefig(CHART_DIR / "occupancy.png", dpi=120); plt.close()
        chart_files.append("occupancy.png")

    # 5. 速度-时间曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    for tid, group in track_df.groupby("车辆ID"):
        group = group.sort_values("帧")
        times = group["时间"].values
        ys = group["中心点Y"].values
        if len(times) < 3:
            continue
        v = np.diff(ys) / (np.diff(times) * pixel_per_meter) * speed_calib * 3.6
        win = min(7, len(v))
        if win > 1:
            v = np.convolve(v, np.ones(win) / win, mode="same")
        t_mid = (times[:-1] + times[1:]) / 2
        n = min(len(t_mid), len(v))
        ax.plot(t_mid[:n], np.abs(v[:n]), linewidth=0.6, alpha=0.6)
    avg = summary.get("平均车速_kmh", 0)
    if avg > 0:
        ax.axhline(y=avg, color="red", linestyle="--", linewidth=1.5,
                   label=f"平均: {avg:.2f} km/h")
        ax.legend(loc="upper right")
    ax.set_title("车辆速度-时间曲线", fontsize=13, fontweight="bold")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("速度 (km/h)")
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout(); plt.savefig(CHART_DIR / "speed_time.png", dpi=120); plt.close()
    chart_files.append("speed_time.png")

    # 6. 车型饼图
    class_counts = summary.get("车型统计", {})
    if class_counts:
        color_map = {"小型车": "#2E86AB", "摩托车": "#A23B72",
                     "中型车": "#C73E1D", "大型车": "#1B998B"}
        colors = [color_map.get(c, "#888") for c in class_counts.keys()]
        fig, ax = plt.subplots(figsize=(9, 7))
        wedges, texts, autotexts = ax.pie(
            class_counts.values(), labels=class_counts.keys(), colors=colors,
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * sum(class_counts.values()) / 100))}辆)",
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=12))
        for at in autotexts:
            at.set_color("white"); at.set_fontweight("bold")
        ax.set_title(f"各车型总数（总计：{sum(class_counts.values())} 辆）",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(); plt.savefig(CHART_DIR / "class_pie.png", dpi=120); plt.close()
        chart_files.append("class_pie.png")

    # 7. 综合对比
    if lane_ids:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].bar([f"车道{l}" for l in lane_ids],
                    [summary["分车道"][l]["平均速度_kmh"] for l in lane_ids], color="#1f77b4")
        axes[0].set_title("平均速度"); axes[0].set_ylabel("km/h"); axes[0].grid(axis="y", alpha=0.3)
        axes[1].bar([f"车道{l}" for l in lane_ids],
                    [summary["分车道"][l]["密度_辆每米"] for l in lane_ids], color="#942654")
        axes[1].set_title("密度"); axes[1].set_ylabel("辆/m"); axes[1].grid(axis="y", alpha=0.3)
        axes[2].bar([f"车道{l}" for l in lane_ids],
                    [summary["分车道"][l]["流量_辆每小时"] for l in lane_ids], color="#ff8800")
        axes[2].set_title("流量"); axes[2].set_ylabel("辆/h"); axes[2].grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(CHART_DIR / "combo.png", dpi=120); plt.close()
        chart_files.append("combo.png")

    return chart_files


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json()
    state["roi"] = data.get("roi", [])
    state["dline"] = data.get("dline", [])
    state["lane_lines"] = data.get("lane_lines", [])
    state["config"] = data.get("config", {})
    state["processing"] = True
    state["progress"] = 0
    state["result"] = None
    state["track_df"] = None
    state["summary"] = None
    state["video_result"] = None
    state["charts"] = []
    state["error"] = None

    def worker():
        try:
            from detector import VehicleDetector
            model_path = resource_path("yolov8m.pt")
            if not os.path.exists(model_path):
                model_path = "yolov8m.pt"
            device = "mps" if platform.system() == "Darwin" else "cpu"
            print(f"[worker] 使用设备: {device}")
            detector = VehicleDetector(model_path, device=device)

            def on_progress(cur, total):
                state["progress"] = int(cur / total * 100)

            passed_df, video_path = detector.process(
                state["video_path"],
                state["roi"], state["dline"], state["lane_lines"],
                state["config"], progress_cb=on_progress)

            detail_csv = OUTPUT_DIR / "detail.csv"
            track_df = pd.read_csv(detail_csv, encoding="utf-8-sig") if detail_csv.exists() else pd.DataFrame()

            summary, passed_df = compute_statistics(passed_df, track_df, state["config"])
            charts = generate_charts(passed_df, track_df, summary, state["config"])

            state["result"] = passed_df
            state["track_df"] = track_df
            state["summary"] = summary
            state["video_result"] = video_path
            state["charts"] = charts
        except Exception as e:
            import traceback
            traceback.print_exc()
            state["error"] = str(e)
        finally:
            state["processing"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/progress")
def progress():
    return jsonify({"progress": state["progress"],
                    "processing": state["processing"],
                    "error": state["error"]})


@app.route("/result")
def result():
    if state["error"]:
        return jsonify({"ok": False, "msg": state["error"]})
    if state["summary"] is None:
        return jsonify({"ok": False, "msg": "还没有结果"})
    return jsonify({"ok": True, "summary": state["summary"],
                    "video_url": "/result_video", "charts": state["charts"]})


@app.route("/chart/<name>")
def chart(name):
    path = CHART_DIR / name
    if not path.exists():
        return "未找到", 404
    return send_file(str(path), mimetype="image/png")


@app.route("/result_video")
def result_video():
    if not state["video_result"] or not os.path.exists(state["video_result"]):
        return "未找到", 404
    return send_file(state["video_result"], mimetype="video/mp4")


@app.route("/download_csv")
def download_csv():
    csv_path = OUTPUT_DIR / "passed.csv"
    if not csv_path.exists():
        return "未找到", 404
    return send_file(str(csv_path), as_attachment=True, download_name="车辆统计.csv")


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import webview

    print("=" * 60)
    print("车辆检测系统 正在启动...")
    print(f"工作目录: {WORK_DIR}")
    print("=" * 60)

    def run_flask():
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(2)

    webview.create_window("车辆检测系统", "http://127.0.0.1:5000",
                          width=1400, height=900, resizable=True)
    webview.start()
