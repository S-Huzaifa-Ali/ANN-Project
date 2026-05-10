/**
 * NutriScan AI — Frontend Application Logic
 * Handles image upload, API calls, bounding box rendering,
 * prediction selection, and nutrition display.
 */

// ===== State =====
const state = {
    file: null,
    detections: [],       // Array of detection results from API
    selectedFoods: {},    // { detectionId: { name, portion_multiplier, nutrition } }
    imageWidth: 0,
    imageHeight: 0,
    originalImage: null,  // base64
};

// ===== DOM Elements =====
const $ = (sel) => document.querySelector(sel);
const uploadZone = $("#upload-zone");
const uploadContent = $("#upload-zone-content");
const uploadPreview = $("#upload-preview");
const previewImage = $("#preview-image");
const fileInput = $("#file-input");
const browseBtn = $("#browse-btn");
const clearBtn = $("#clear-btn");
const analyzeBtn = $("#analyze-btn");
const loadingOverlay = $("#loading-overlay");
const errorBanner = $("#error-banner");
const errorText = $("#error-text");
const errorClose = $("#error-close");
const resultsSection = $("#results-section");
const resultsMessage = $("#results-message");
const detectionCanvas = $("#detection-canvas");
const detectionsGrid = $("#detections-grid");
const nutritionSection = $("#nutrition-section");
const nutritionSummaryCards = $("#nutrition-summary-cards");
const nutritionTableBody = $("#nutrition-table-body");
const nutritionTableFoot = $("#nutrition-table-foot");

// ===== Upload Handling =====

// Drag & Drop
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files[0]);
});

// Click to browse
uploadZone.addEventListener("click", (e) => {
    if (e.target.closest("#clear-btn")) return;
    fileInput.click();
});
browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFileSelect(fileInput.files[0]);
});

// Clear button
clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetState();
});

// Analyze button
analyzeBtn.addEventListener("click", () => {
    if (state.file) analyzeImage();
});

// Error close
errorClose.addEventListener("click", () => {
    errorBanner.style.display = "none";
});

/**
 * Handle file selection — validate and show preview.
 */
function handleFileSelect(file) {
    const validTypes = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
    if (!validTypes.includes(file.type)) {
        showError("Invalid file type. Please upload PNG, JPG, WEBP, or BMP.");
        return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showError("File too large. Maximum size is 16MB.");
        return;
    }

    state.file = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadContent.style.display = "none";
        uploadPreview.style.display = "flex";
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);

    // Hide previous results
    resultsSection.style.display = "none";
    nutritionSection.style.display = "none";
    errorBanner.style.display = "none";
}

/**
 * Reset to initial state.
 */
function resetState() {
    state.file = null;
    state.detections = [];
    state.selectedFoods = {};
    state.originalImage = null;
    fileInput.value = "";
    uploadContent.style.display = "";
    uploadPreview.style.display = "none";
    previewImage.src = "";
    analyzeBtn.disabled = true;
    resultsSection.style.display = "none";
    nutritionSection.style.display = "none";
    errorBanner.style.display = "none";
}

// ===== API Call =====

/**
 * Send the image to the backend for analysis.
 */
async function analyzeImage() {
    if (!state.file) return;

    showLoading(true);
    errorBanner.style.display = "none";

    const formData = new FormData();
    formData.append("image", state.file);

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Server error occurred.");
        }

        if (!data.success) {
            throw new Error(data.error || "Analysis failed.");
        }

        // Store results
        state.detections = data.detections || [];
        state.imageWidth = data.image_width;
        state.imageHeight = data.image_height;
        state.originalImage = data.original_image;

        // Initialize selected foods (default to top prediction)
        state.selectedFoods = {};
        state.detections.forEach((det) => {
            if (det.predictions && det.predictions.length > 0) {
                state.selectedFoods[det.id] = {
                    name: det.predictions[0].class_name,
                    portion_multiplier: 1.0,
                    nutrition: det.nutrition,
                };
            }
        });

        // Render results
        renderResults(data.message);
        showLoading(false);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
        showLoading(false);
        showError(err.message);
    }
}

// ===== Rendering =====

/**
 * Render all detection results.
 */
function renderResults(message) {
    resultsMessage.textContent = message;
    resultsSection.style.display = "";

    if (state.detections.length === 0) {
        detectionsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align:center; padding:2rem; color:var(--text-muted);">
                No food items were detected. Try a different image.
            </div>`;
        nutritionSection.style.display = "none";
        drawDetectionCanvas();
        return;
    }

    drawDetectionCanvas();
    renderDetectionCards();
    renderNutrition();
}

/**
 * Draw bounding boxes on the canvas.
 */
function drawDetectionCanvas() {
    const ctx = detectionCanvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
        // Scale canvas to fit container while maintaining aspect ratio
        const wrapper = document.getElementById("detection-canvas-wrapper");
        const maxWidth = wrapper.clientWidth - 32; // padding
        const scale = Math.min(1, maxWidth / img.width);
        const displayW = Math.round(img.width * scale);
        const displayH = Math.round(img.height * scale);

        detectionCanvas.width = displayW;
        detectionCanvas.height = displayH;

        // Draw image
        ctx.drawImage(img, 0, 0, displayW, displayH);

        // Draw bounding boxes
        const colors = ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899"];

        state.detections.forEach((det, i) => {
            const color = colors[i % colors.length];
            const [x1, y1, x2, y2] = det.bbox;
            const sx = x1 * scale, sy = y1 * scale;
            const sw = (x2 - x1) * scale, sh = (y2 - y1) * scale;

            // Draw box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(sx, sy, sw, sh);

            // Draw label background
            const selected = state.selectedFoods[det.id];
            const label = selected ? selected.name : `Item ${i + 1}`;
            ctx.font = `bold ${Math.max(12, 14 * scale)}px Inter, sans-serif`;
            const textW = ctx.measureText(label).width + 12;
            const textH = Math.max(18, 22 * scale);

            ctx.fillStyle = color;
            ctx.fillRect(sx, sy - textH, textW, textH);

            // Draw label text
            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, sx + 6, sy - 5);
        });
    };
    img.src = "data:image/jpeg;base64," + state.originalImage;
}

/**
 * Render detection cards with top-3 prediction selection.
 */
function renderDetectionCards() {
    detectionsGrid.innerHTML = "";

    state.detections.forEach((det, i) => {
        const card = document.createElement("div");
        card.className = "detection-card";
        card.style.animationDelay = `${i * 0.1}s`;

        const selected = state.selectedFoods[det.id];

        // Header with crop thumbnail
        let headerHTML = `
            <div class="detection-card-header">
                <img class="detection-card-thumb" src="data:image/jpeg;base64,${det.crop_image}" alt="Detected food ${i + 1}">
                <div class="detection-card-info">
                    <div class="detection-card-title">Food Item #${i + 1}</div>
                    <div class="detection-card-conf">Detection confidence: ${(det.detection_confidence * 100).toFixed(1)}%</div>
                </div>
            </div>`;

        // Predictions (top-3 selectable)
        let predictionsHTML = `<div class="prediction-label">Select the correct food (Top 3 Predictions)</div>`;

        det.predictions.forEach((pred, j) => {
            const isSelected = selected && selected.name === pred.class_name;
            const confPercent = (pred.confidence * 100).toFixed(1);

            predictionsHTML += `
                <div class="prediction-option ${isSelected ? "selected" : ""}"
                     data-det-id="${det.id}" data-pred-index="${j}"
                     data-food-name="${pred.class_name}" id="pred-${det.id}-${j}">
                    <div class="prediction-radio"></div>
                    <span class="prediction-name">${pred.class_name}</span>
                    <div class="prediction-conf-bar">
                        <div class="prediction-conf-fill" style="width:${confPercent}%"></div>
                    </div>
                    <span class="prediction-conf-value">${confPercent}%</span>
                </div>`;
        });

        // Portion slider
        const currentPortion = selected ? selected.portion_multiplier : 1.0;
        const portionLabels = { 0.5: "Small", 1.0: "Medium", 1.5: "Large", 2.0: "Extra Large" };
        const portionLabel = portionLabels[currentPortion] || `${currentPortion}x`;

        let portionHTML = `
            <div class="portion-control">
                <div class="portion-label">
                    <span>Portion Size</span>
                    <span id="portion-val-${det.id}">${portionLabel} (${currentPortion}x)</span>
                </div>
                <input type="range" class="portion-slider" id="portion-slider-${det.id}"
                       min="0.5" max="2.0" step="0.5" value="${currentPortion}"
                       data-det-id="${det.id}">
            </div>`;

        card.innerHTML = headerHTML + `<div class="detection-card-body">${predictionsHTML}${portionHTML}</div>`;
        detectionsGrid.appendChild(card);
    });

    // Attach event listeners for prediction selection
    document.querySelectorAll(".prediction-option").forEach((el) => {
        el.addEventListener("click", handlePredictionSelect);
    });

    // Attach event listeners for portion sliders
    document.querySelectorAll(".portion-slider").forEach((el) => {
        el.addEventListener("input", handlePortionChange);
    });
}

/**
 * Handle user selecting a prediction for a detection.
 */
async function handlePredictionSelect(e) {
    const option = e.currentTarget;
    const detId = parseInt(option.dataset.detId);
    const foodName = option.dataset.foodName;

    // Update selection UI
    document.querySelectorAll(`.prediction-option[data-det-id="${detId}"]`).forEach((el) => {
        el.classList.remove("selected");
    });
    option.classList.add("selected");

    // Get current portion multiplier
    const currentPortion = state.selectedFoods[detId] ? state.selectedFoods[detId].portion_multiplier : 1.0;

    // Fetch nutrition for the newly selected food
    const nutrition = await fetchNutrition(foodName, currentPortion);

    // Update state
    state.selectedFoods[detId] = {
        name: foodName,
        portion_multiplier: currentPortion,
        nutrition: nutrition,
    };

    // Re-render nutrition summary and canvas labels
    renderNutrition();
    drawDetectionCanvas();
}

/**
 * Handle portion slider change.
 */
async function handlePortionChange(e) {
    const slider = e.currentTarget;
    const detId = parseInt(slider.dataset.detId);
    const portion = parseFloat(slider.value);

    // Update label
    const portionLabels = { 0.5: "Small", 1.0: "Medium", 1.5: "Large", 2.0: "Extra Large" };
    const label = portionLabels[portion] || `${portion}x`;
    const valEl = document.getElementById(`portion-val-${detId}`);
    if (valEl) valEl.textContent = `${label} (${portion}x)`;

    // Get current food name
    const current = state.selectedFoods[detId];
    if (!current) return;

    // Fetch updated nutrition
    const nutrition = await fetchNutrition(current.name, portion);

    // Update state
    state.selectedFoods[detId] = {
        name: current.name,
        portion_multiplier: portion,
        nutrition: nutrition,
    };

    // Re-render nutrition
    renderNutrition();
}

/**
 * Fetch nutrition data from the API.
 */
async function fetchNutrition(foodName, portionMultiplier) {
    try {
        const response = await fetch("/api/nutrition", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                food_name: foodName,
                portion_multiplier: portionMultiplier,
            }),
        });

        const data = await response.json();
        if (data.success && data.nutrition) {
            return data.nutrition;
        }
        return null;
    } catch {
        return null;
    }
}

/**
 * Render the nutrition summary section.
 */
function renderNutrition() {
    const foods = Object.values(state.selectedFoods).filter((f) => f.nutrition);

    if (foods.length === 0) {
        nutritionSection.style.display = "none";
        return;
    }

    nutritionSection.style.display = "";

    // Calculate totals
    let totalCal = 0, totalProtein = 0, totalCarbs = 0, totalFat = 0;
    foods.forEach((f) => {
        if (f.nutrition) {
            totalCal += f.nutrition.calories || 0;
            totalProtein += f.nutrition.protein_g || 0;
            totalCarbs += f.nutrition.carbs_g || 0;
            totalFat += f.nutrition.fat_g || 0;
        }
    });

    // Summary cards
    nutritionSummaryCards.innerHTML = `
        <div class="nutri-card calories">
            <div class="nutri-card-value">${Math.round(totalCal)}</div>
            <div class="nutri-card-unit">kcal</div>
            <div class="nutri-card-label">Calories</div>
        </div>
        <div class="nutri-card protein">
            <div class="nutri-card-value">${totalProtein.toFixed(1)}</div>
            <div class="nutri-card-unit">grams</div>
            <div class="nutri-card-label">Protein</div>
        </div>
        <div class="nutri-card carbs">
            <div class="nutri-card-value">${totalCarbs.toFixed(1)}</div>
            <div class="nutri-card-unit">grams</div>
            <div class="nutri-card-label">Carbs</div>
        </div>
        <div class="nutri-card fat">
            <div class="nutri-card-value">${totalFat.toFixed(1)}</div>
            <div class="nutri-card-unit">grams</div>
            <div class="nutri-card-label">Fat</div>
        </div>`;

    // Detail table
    nutritionTableBody.innerHTML = "";
    foods.forEach((f) => {
        if (!f.nutrition) return;
        const n = f.nutrition;
        const portionLabels = { 0.5: "Small", 1.0: "Medium", 1.5: "Large", 2.0: "Extra Large" };
        const portionLabel = portionLabels[f.portion_multiplier] || `${f.portion_multiplier}x`;
        const weightInfo = n.estimated_weight_g ? ` (~${n.estimated_weight_g}g)` : "";

        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${n.food_item}</td>
            <td>${portionLabel}${weightInfo}</td>
            <td>${n.calories} kcal</td>
            <td>${n.protein_g}g</td>
            <td>${n.carbs_g}g</td>
            <td>${n.fat_g}g</td>`;
        nutritionTableBody.appendChild(row);
    });

    // Table footer totals
    nutritionTableFoot.innerHTML = `
        <tr>
            <td colspan="2">Total</td>
            <td>${Math.round(totalCal)} kcal</td>
            <td>${totalProtein.toFixed(1)}g</td>
            <td>${totalCarbs.toFixed(1)}g</td>
            <td>${totalFat.toFixed(1)}g</td>
        </tr>`;
}

// ===== Utility =====

function showLoading(show) {
    loadingOverlay.style.display = show ? "flex" : "none";
}

function showError(message) {
    errorText.textContent = message;
    errorBanner.style.display = "flex";
}

// ===== Nav Highlighting =====
document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
        document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
        link.classList.add("active");
    });
});
