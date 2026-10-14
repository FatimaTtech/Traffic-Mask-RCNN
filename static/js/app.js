/**
 * Traffic Mask R-CNN Frontend Controller
 * Handles model metadata fetch, drag-and-drop uploads, sample selection,
 * inference execution, side-by-side rendering, and interactive controls.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State variables
    let selectedFile = null;
    let selectedSample = null;
    let modelMetadata = null;
    let isProcessing = false;

    // DOM Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const selectedFileName = document.getElementById("selectedFileName");
    const detectBtn = document.getElementById("detectBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    const sampleChips = document.getElementById("sampleChips");

    const confidenceSlider = document.getElementById("confidenceSlider");
    const confidenceValue = document.getElementById("confidenceValue");
    const opacitySlider = document.getElementById("opacitySlider");
    const opacityValue = document.getElementById("opacityValue");

    const toggleMasks = document.getElementById("toggleMasks");
    const toggleBoxes = document.getElementById("toggleBoxes");
    const toggleLabels = document.getElementById("toggleLabels");

    const alertBanner = document.getElementById("alertBanner");
    const alertMessage = document.getElementById("alertMessage");

    const emptyState = document.getElementById("emptyState");
    const resultsContainer = document.getElementById("resultsContainer");

    const originalImagePreview = document.getElementById("originalImagePreview");
    const annotatedImagePreview = document.getElementById("annotatedImagePreview");
    const downloadAnnotatedBtn = document.getElementById("downloadAnnotatedBtn");

    const metricTotalDetections = document.getElementById("metricTotalDetections");
    const metricInferenceTime = document.getElementById("metricInferenceTime");
    const metricResolution = document.getElementById("metricResolution");
    const metricSource = document.getElementById("metricSource");
    const classPillsRow = document.getElementById("classPillsRow");
    const detectionsTableBody = document.getElementById("detectionsTableBody");
    const tableItemCount = document.getElementById("tableItemCount");

    const modelStatusDetails = document.getElementById("modelStatusDetails");
    const modelStatusBadge = document.getElementById("modelStatusBadge");
    const classLegend = document.getElementById("classLegend");

    // =========================================================================
    // 1. Initialize & Fetch Model Metadata
    // =========================================================================
    async function initModelInfo() {
        try {
            const resp = await fetch("/api/model_info");
            const data = await resp.json();
            modelMetadata = data;

            if (data.model_ready) {
                const lossText = data.loss ? ` • Loss ${data.loss.toFixed(4)}` : "";
                const epochText = data.epoch ? `Epoch ${data.epoch}` : "Trained";
                modelStatusDetails.textContent = `${epochText}${lossText} • ${data.device.toUpperCase()}`;
            } else {
                modelStatusDetails.textContent = "Model Error: Checkpoint missing";
                const dot = modelStatusBadge.querySelector(".status-dot");
                if (dot) dot.classList.replace("pulsing", "error");
                showAlert(`Model failed to initialize: ${data.error || "Missing checkpoint"}`, "error");
            }

            // Populate class legend
            if (data.classes && data.class_colors) {
                classLegend.innerHTML = "";
                data.classes.forEach(cls => {
                    if (cls === "background") return;
                    const color = data.class_colors[cls] || "#3b82f6";
                    const chip = document.createElement("div");
                    chip.className = "legend-chip";
                    chip.innerHTML = `
                        <span class="legend-color-box" style="background-color: ${color}"></span>
                        <span>${cls}</span>
                    `;
                    classLegend.appendChild(chip);
                });
            }
        } catch (err) {
            console.error("Failed fetching model metadata:", err);
            modelStatusDetails.textContent = "Offline / Server Error";
        }
    }

    // =========================================================================
    // 2. Alert Banner Helpers
    // =========================================================================
    function showAlert(msg, type = "error") {
        alertMessage.textContent = msg;
        alertBanner.className = `alert-banner ${type === "info" ? "info" : ""}`;
        alertBanner.classList.remove("hidden");
    }

    window.dismissAlert = function() {
        alertBanner.classList.add("hidden");
    };

    // =========================================================================
    // 3. Slider & Toggle Events
    // =========================================================================
    confidenceSlider.addEventListener("input", (e) => {
        confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    opacitySlider.addEventListener("input", (e) => {
        opacityValue.textContent = `${e.target.value}%`;
    });

    // =========================================================================
    // 4. File Upload & Drag-and-Drop Handling
    // =========================================================================
    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    function handleSelectedFile(file) {
        // Validate MIME type / extension
        const validExtensions = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
        if (!validExtensions.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp|bmp)$/i)) {
            showAlert("Please upload a valid image file (JPG, PNG, WEBP, or BMP).");
            return;
        }

        selectedFile = file;
        selectedSample = null;

        // Clear active state on sample buttons
        document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));

        selectedFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedFileName.classList.remove("hidden");
        detectBtn.disabled = false;
        dismissAlert();
    }

    // =========================================================================
    // 5. Sample Chips Selection
    // =========================================================================
    sampleChips.addEventListener("click", (e) => {
        const btn = e.target.closest(".sample-btn");
        if (!btn) return;

        const sampleName = btn.dataset.sample;
        selectedSample = sampleName;
        selectedFile = null;
        fileInput.value = "";

        document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        selectedFileName.textContent = `Selected: ${sampleName}`;
        selectedFileName.classList.remove("hidden");
        detectBtn.disabled = false;
        dismissAlert();
    });

    // =========================================================================
    // 6. Run Mask R-CNN Detection
    // =========================================================================
    detectBtn.addEventListener("click", async () => {
        if (isProcessing) return;
        if (!selectedFile && !selectedSample) {
            showAlert("Please select or upload an image first.");
            return;
        }

        setLoading(true);
        dismissAlert();

        const formData = new FormData();
        if (selectedFile) {
            formData.append("image", selectedFile);
        } else if (selectedSample) {
            formData.append("sample_name", selectedSample);
        }

        formData.append("confidence", confidenceSlider.value);
        formData.append("mask_opacity", (parseFloat(opacitySlider.value) / 100).toFixed(2));
        formData.append("draw_boxes", toggleBoxes.checked);
        formData.append("draw_masks", toggleMasks.checked);
        formData.append("draw_labels", toggleLabels.checked);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || `Server responded with status ${response.status}`);
            }

            renderResults(data);

        } catch (err) {
            console.error("Detection error:", err);
            showAlert(err.message || "An unexpected error occurred during prediction.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(loading) {
        isProcessing = loading;
        detectBtn.disabled = loading;
        if (loading) {
            btnText.textContent = "Running Mask R-CNN...";
            btnSpinner.classList.remove("hidden");
        } else {
            btnText.textContent = "Run Mask R-CNN Detection";
            btnSpinner.classList.add("hidden");
        }
    }

    // =========================================================================
    // 7. Render Prediction Results
    // =========================================================================
    function renderResults(data) {
        emptyState.classList.add("hidden");
        resultsContainer.classList.remove("hidden");

        // Metrics
        metricTotalDetections.textContent = data.total_detected;
        metricInferenceTime.textContent = `${data.inference_time_ms} ms`;
        metricResolution.textContent = `${data.width} × ${data.height}`;
        metricSource.textContent = data.source_name;

        // Images
        originalImagePreview.src = data.original_image;
        annotatedImagePreview.src = data.annotated_image;
        downloadAnnotatedBtn.href = data.annotated_image;
        downloadAnnotatedBtn.download = `pred_${data.source_name.replace(/\.[^/.]+$/, "")}.jpg`;

        // Category breakdown pills
        classPillsRow.innerHTML = "";
        const counts = data.class_counts || {};
        let activeCategories = 0;

        for (const [clsName, count] of Object.entries(counts)) {
            if (count > 0) {
                activeCategories++;
                const color = (modelMetadata && modelMetadata.class_colors && modelMetadata.class_colors[clsName]) || "#3b82f6";
                const pill = document.createElement("div");
                pill.className = "count-pill";
                pill.innerHTML = `
                    <span class="legend-color-box" style="background-color: ${color}"></span>
                    <span>${clsName}</span>
                    <span class="count-pill-badge">${count}</span>
                `;
                classPillsRow.appendChild(pill);
            }
        }

        if (activeCategories === 0) {
            const pill = document.createElement("div");
            pill.className = "count-pill";
            pill.innerHTML = `<span>No traffic objects met confidence threshold (${confidenceSlider.value})</span>`;
            classPillsRow.appendChild(pill);
        }

        // Table rows
        detectionsTableBody.innerHTML = "";
        tableItemCount.textContent = `${data.detections.length} items`;

        if (data.detections.length === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="6" style="text-align:center; padding: 24px; color: var(--text-muted)">No objects detected above the selected confidence threshold. Try lowering the threshold slider.</td>`;
            detectionsTableBody.appendChild(tr);
            return;
        }

        data.detections.forEach((det, i) => {
            const tr = document.createElement("tr");
            const boxStr = `[${det.box.join(", ")}]`;
            const color = det.color || "#3b82f6";
            const scorePct = (det.score * 100).toFixed(1);

            tr.innerHTML = `
                <td>${i + 1}</td>
                <td>
                    <span class="class-badge" style="color: ${color}">
                        <span class="legend-color-box" style="background-color: ${color}"></span>
                        ${det.class_name}
                    </span>
                </td>
                <td>
                    <div class="score-cell">
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${scorePct}%; background-color: ${color}"></div>
                        </div>
                        <span>${scorePct}%</span>
                    </div>
                </td>
                <td class="coord-cell">${boxStr}</td>
                <td class="coord-cell">${det.box_area.toLocaleString()} px²</td>
                <td class="coord-cell">${det.mask_pixels.toLocaleString()} px</td>
            `;
            detectionsTableBody.appendChild(tr);
        });

        // Smooth scroll to results viewport if on mobile
        if (window.innerWidth < 900) {
            resultsContainer.scrollIntoView({ behavior: "smooth" });
        }
    }

    // Initialize application
    initModelInfo();
});
