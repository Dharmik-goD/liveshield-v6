
/* V6_TRACKING_CAMERA_SCRIPT */
(() => {
    const video = document.getElementById("trackVideo");
    const canvas = document.getElementById("trackCanvas");
    const startButton = document.getElementById("trackStart");
    const stopButton = document.getElementById("trackStop");
    const resetButton = document.getElementById("trackReset");

    if (!video || !canvas || !startButton || startButton.dataset.initialized) return;
    startButton.dataset.initialized = "true";

    const ctx = canvas.getContext("2d");
    const el = id => document.getElementById(id);

    let stream = null;
    let running = false;
    let busy = false;
    let frameCount = 0;
    let fpsCount = 0;
    let fpsStart = performance.now();

    function status(message) {
        el("trackStatus").textContent = message;
    }

    function drawTracking(data) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const colors = ["#00e5ff", "#00ff9d", "#ffcc66", "#c792ff", "#ff7b9c"];

        for (const object of data.tracked_objects || []) {
            const [x1, y1, x2, y2] = object.box;
            const color = colors[object.track_id % colors.length];

            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2-x1, y2-y1);

            const points = object.trajectory || [];
            if (points.length > 1) {
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                points.forEach((point, index) => {
                    if (index === 0) ctx.moveTo(point[0], point[1]);
                    else ctx.lineTo(point[0], point[1]);
                });
                ctx.stroke();
            }

            const label = "ID " + object.track_id + " ? " + object.label;
            ctx.font = "bold 15px Arial";
            const width = ctx.measureText(label).width + 12;
            const labelY = Math.max(0, y1 - 27);

            ctx.fillStyle = color;
            ctx.fillRect(x1, labelY, width, 25);
            ctx.fillStyle = "#07111e";
            ctx.fillText(label, x1 + 6, labelY + 18);
        }
    }

    function updateResults(data) {
        const objects = data.tracked_objects || [];

        el("trackActive").textContent = data.active_count ?? objects.length;
        el("trackFrames").textContent = frameCount;
        el("trackLatency").textContent = data.inference_latency_ms + " ms";
        el("trackObjectCount").textContent = objects.length + " objects";

        el("trackResults").innerHTML = objects.length
            ? objects.map(object =>
                '<div class="tracking-object">' +
                '<strong>Track #' + object.track_id + ' ? ' + object.label + '</strong>' +
                '<p>Detection score: ' + (object.confidence * 100).toFixed(1) + '%</p>' +
                '<p>Position: ' + object.centroid.join(", ") + '</p>' +
                '<p>Trajectory points: ' + object.trajectory.length + '</p>' +
                '</div>'
              ).join("")
            : '<div class="workspace"><strong>No objects detected</strong><p>Waiting for detectable objects.</p></div>';
    }

    async function processFrame() {
        if (!running || busy) return;
        busy = true;

        try {
            const capture = document.createElement("canvas");
            capture.width = video.videoWidth;
            capture.height = video.videoHeight;
            capture.getContext("2d").drawImage(video, 0, 0);

            const blob = await new Promise(resolve =>
                capture.toBlob(resolve, "image/jpeg", 0.82)
            );

            if (!blob) throw new Error("Camera frame capture failed.");

            const form = new FormData();
            form.append("frame", blob, "tracking-frame.jpg");

            const response = await fetch("/api/v6/tracking/frame", {
                method:"POST",
                body:form
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Tracking request failed.");
            if (!running) return;

            frameCount++;
            fpsCount++;

            canvas.width = data.frame.width;
            canvas.height = data.frame.height;

            drawTracking(data);
            updateResults(data);

            const now = performance.now();
            if (now - fpsStart >= 1000) {
                el("trackFPS").textContent =
                    Math.round(fpsCount * 1000 / (now - fpsStart));
                fpsCount = 0;
                fpsStart = now;
            }

            status("Tracking active ? " + data.device);

        } catch (error) {
            running = false;
            startButton.disabled = false;
            stopButton.disabled = true;
            status("Tracking error: " + error.message);
        } finally {
            busy = false;
        }

        if (running) setTimeout(processFrame, 120);
    }

    async function stopCamera() {
        running = false;

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        video.srcObject = null;
        video.style.display = "none";
        el("trackEmpty").style.display = "block";

        startButton.disabled = false;
        stopButton.disabled = true;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        status("Camera stopped ? Tracking inactive");
    }

    startButton.addEventListener("click", async () => {
        try {
            startButton.disabled = true;
            status("Requesting camera permission...");

            stream = await navigator.mediaDevices.getUserMedia({
                video:true,
                audio:false
            });

            video.srcObject = stream;
            await video.play();

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            video.style.display = "block";
            el("trackEmpty").style.display = "none";

            const reset = await fetch("/api/v6/tracking/reset", {method:"POST"});
            if (!reset.ok) throw new Error("Could not reset tracking session.");

            frameCount = 0;
            fpsCount = 0;
            fpsStart = performance.now();

            el("trackFrames").textContent = "0";
            el("trackFPS").textContent = "0";

            running = true;
            stopButton.disabled = false;
            status("Starting tracking...");
            processFrame();

        } catch (error) {
            await stopCamera();
            status("Camera error: " + error.message);
        }
    });

    stopButton.addEventListener("click", stopCamera);

    resetButton.addEventListener("click", async () => {
        try {
            const response = await fetch("/api/v6/tracking/reset", {method:"POST"});
            if (!response.ok) throw new Error("Reset failed.");

            frameCount = 0;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            el("trackActive").textContent = "0";
            el("trackFrames").textContent = "0";
            el("trackLatency").textContent = "-- ms";
            el("trackFPS").textContent = "0";
            el("trackObjectCount").textContent = "0 objects";
            el("trackResults").innerHTML =
                '<div class="workspace"><strong>Tracking session reset</strong><p>Ready for a fresh session.</p></div>';

            status(running ? "Tracking active ? Session reset" : "Tracking session reset");
        } catch (error) {
            status("Reset error: " + error.message);
        }
    });

    window.addEventListener("beforeunload", () => {
        if (stream) stream.getTracks().forEach(track => track.stop());
    });
})();
