// Dashboard JavaScript for Real-Time Crowd Detection System

// Global variables
let crowdTrendsChart = null;
let performanceChart = null;
let roiCanvas = null;
let roiContext = null;
let roiPoints = [];
let currentCameraId = null;
let cameraImage = null;
let roiTemplates = null;
let userInteracted = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard.js loaded');
    
    // Add canvas container styles
    const style = document.createElement('style');
    style.textContent = `
        .canvas-container {
            position: relative;
            width: 100%;
            height: auto;
            min-height: 300px;
            background: #f8f9fa;
            overflow: hidden;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
        
        #roi-canvas {
            max-width: 100%;
            height: auto;
            display: block;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }
        
        #canvas-loading-indicator {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .roi-point {
            position: absolute;
            width: 10px;
            height: 10px;
            background: red;
            border: 2px solid white;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
        
        .roi-line {
            position: absolute;
            background: rgba(255, 0, 0, 0.5);
            height: 2px;
            pointer-events: none;
        }
    `;
    document.head.appendChild(style);
    
    // Initialize charts
    initCharts();
    
    // Load cameras
    loadCameras();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize save threshold button
    const saveThresholdBtn = document.getElementById('save-threshold-btn');
    if (saveThresholdBtn) {
        saveThresholdBtn.addEventListener('click', saveThreshold);
    }
    
    // Add event listener for ROI modal shown
    const roiModal = document.getElementById('roiModal');
    if (roiModal) {
        // Store the last focused element before opening modal
        let lastFocusedElement = null;

        roiModal.addEventListener('show.bs.modal', function() {
            // Store the currently focused element
            lastFocusedElement = document.activeElement;
        });

        roiModal.addEventListener('shown.bs.modal', function() {
            // Remove aria-hidden and use inert for better accessibility
            this.removeAttribute('aria-hidden');
            this.setAttribute('inert', '');
            
            // Ensure canvas is initialized
            const canvas = document.getElementById('roi-canvas');
            if (canvas) {
                canvas.willReadFrequently = true;
                roiCanvas = canvas;
                roiContext = canvas.getContext('2d', { 
                    willReadFrequently: true,
                    alpha: false
                });
                
                // Set up canvas event listeners
                canvas.addEventListener('click', function(event) {
                    addRoiPoint(event);
                });
                
                // Initialize ROI template handling
                initializeRoiTemplateHandling();
            }

            // Focus the first focusable element in the modal
            const focusableElements = this.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            if (focusableElements.length > 0) {
                focusableElements[0].focus();
            }
        });
        
        // Add hide event listener to clean up
        roiModal.addEventListener('hide.bs.modal', function() {
            // Remove focus from any elements inside the modal
            const focusedElement = document.activeElement;
            if (focusedElement && this.contains(focusedElement)) {
                focusedElement.blur();
            }
        });

        roiModal.addEventListener('hidden.bs.modal', function() {
            // Remove inert attribute
            this.removeAttribute('inert');
            
            // Clear ROI points when modal is closed
            roiPoints = [];
            
            // Clear the canvas
            if (roiCanvas && roiContext) {
                roiContext.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
            }

            // Restore focus to the element that was focused before opening the modal
            if (lastFocusedElement) {
                lastFocusedElement.focus();
                lastFocusedElement = null;
            }
        });

        // Handle escape key to close modal
        roiModal.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                const focusedElement = document.activeElement;
                if (focusedElement && this.contains(focusedElement)) {
                    focusedElement.blur();
                }
            }
        });
    }
    
    // Start data refresh
    setInterval(refreshData, 2000);
    
    // Set threshold event handlers
    function setupThresholdEvents() {
        // Set threshold button click
        document.querySelectorAll('.set-threshold-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const cameraId = this.getAttribute('data-camera-id');
                openThresholdModal(cameraId);
            });
        });
        
        // Save threshold button
        const saveThresholdBtn = document.getElementById('save-threshold-btn');
        if (saveThresholdBtn) {
            saveThresholdBtn.addEventListener('click', saveThreshold);
        }
    }

    // Open threshold modal
    function openThresholdModal(cameraId) {
        document.getElementById('threshold-camera-id').value = cameraId;
        
        // Get current threshold values
        const thresholdSpan = document.getElementById(`threshold-${cameraId}`);
        const currentThreshold = thresholdSpan ? thresholdSpan.textContent.split(': ')[1] : '50';
        
        // Get current density threshold
        const densityThresholdSpan = document.getElementById(`density-threshold-${cameraId}`);
        const currentDensityThreshold = densityThresholdSpan ? densityThresholdSpan.textContent.split(': ')[1] : '20';
        
        // Get current priority
        const prioritySpan = document.getElementById(`priority-${cameraId}`);
        const currentPriority = prioritySpan ? prioritySpan.textContent.split(': ')[1] : '3';
        
        // Set current values in form
        document.getElementById('alert-threshold').value = currentThreshold;
        document.getElementById('threshold-density').value = currentDensityThreshold;
        document.getElementById('camera-priority').value = currentPriority;
        
        // Show modal
        const thresholdModal = new bootstrap.Modal(document.getElementById('thresholdModal'));
        thresholdModal.show();
    }

    // Save threshold
    function saveThreshold() {
        const cameraId = document.getElementById('threshold-camera-id').value;
        const threshold = parseInt(document.getElementById('alert-threshold').value, 10);
        const densityThreshold = parseInt(document.getElementById('threshold-density').value, 10);
        const priority = parseInt(document.getElementById('camera-priority').value, 10);

        if (isNaN(threshold) || isNaN(densityThreshold) || isNaN(priority)) {
            alert('Please enter valid numbers for all fields (Threshold, Density Threshold, Priority).');
            return;
        }

        fetch('/api/set_threshold', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                id: cameraId,
                threshold: threshold,
                density_threshold: densityThreshold,
                priority: priority
            })
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('thresholdModal'));
                modal.hide();
                // Always reload cameras to get latest data
                loadCameras();
                alert('Camera settings updated successfully');
            } else {
                alert('Error setting thresholds: ' + (result.message || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error setting thresholds:', error);
            alert('Error setting thresholds');
        });
    }

    // Helper function to get priority badge class
    function getPriorityBadgeClass(priority) {
        switch(priority) {
            case 1: return 'bg-secondary';
            case 2: return 'bg-info';
            case 3: return 'bg-primary';
            case 4: return 'bg-warning';
            case 5: return 'bg-danger';
            case 6: return 'bg-dark';
            default: return 'bg-secondary';
        }
    }

    // Initialize event handlers
    function initializeEvents() {
        // Call setupThresholdEvents when page loads
        setupThresholdEvents();
        
        // Watch for dynamic content changes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length) {
                    // Re-attach threshold events when new content is added
                    setupThresholdEvents();
                }
            });
        });
        
        // Start observing the camera container for DOM changes
        const cameraContainer = document.getElementById('camera-container');
        if (cameraContainer) {
            observer.observe(cameraContainer, { childList: true, subtree: true });
        }
    }

    // Initialize
    initializeEvents();

    // Add event listener for user interaction
    window.addEventListener('click', () => userInteracted = true, { once: true });
});

// Initialize charts
function initCharts() {
    // Crowd trends chart
    const crowdCtx = document.getElementById('crowdTrendsChart').getContext('2d');
    crowdTrendsChart = new Chart(crowdCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Crowd Count'
                    },
                    beginAtZero: true
                }
            }
        }
    });
    
    // Performance chart
    const perfCtx = document.getElementById('fpsChart').getContext('2d');
    performanceChart = new Chart(perfCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'FPS',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
                {
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Value'
                    }
                }
            }
        }
    });
}

// Set up event listeners
function setupEventListeners() {
    // Add camera button
    document.getElementById('save-camera-btn').addEventListener('click', addCamera);
    
    // ROI canvas setup
    const canvas = document.getElementById('roi-canvas');
    if (canvas) {
        roiCanvas = canvas;
        roiContext = canvas.getContext('2d', {
            willReadFrequently: true,
            alpha: false
        });
        
        // ROI canvas click
        canvas.addEventListener('click', function(event) {
            addRoiPoint(event);
        });
        
        // Clear ROI button
        document.getElementById('clear-roi-btn').addEventListener('click', clearRoiPoints);
        
        // Save ROI button
        document.getElementById('save-roi-btn').addEventListener('click', saveRoi);
    }
}

// Load cameras from API
function loadCameras() {
    fetch('/api/cameras')
        .then(response => response.json())
        .then(cameras => {
            renderCameraCards(cameras);
        })
        .catch(error => {
            console.error('Error loading cameras:', error);
        });
}

// Render camera cards
function renderCameraCards(cameras) {
    const container = document.getElementById('camera-feed-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Check if cameras is empty
    if (Object.keys(cameras).length === 0) {
        container.innerHTML = '<div class="col-12"><div class="alert alert-info">No cameras added yet. Click the "Add Camera" button to add a camera.</div></div>';
        return;
    }
    
    // Create card for each camera
    for (const [cameraId, camera] of Object.entries(cameras)) {
        const card = document.createElement('div');
        card.className = 'col-md-6 col-lg-4';
        card.innerHTML = `
            <div id="camera-card-${cameraId}" class="camera-card">
                <div class="camera-feed-container position-relative">
                    <img src="/video_feed/${cameraId}" class="camera-feed" alt="${camera.name}">
                    <span class="position-absolute badge ${priorityBadgeClass}" style="top: 10px; right: 10px;">
                        Priority: ${priorityText}
                    </span>
                </div>
                <div class="camera-info p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5>${camera.name}</h5>
                        <div>
                            <button class="btn btn-sm btn-outline-primary set-roi-btn" data-camera-id="${cameraId}">
                                <i class="bi bi-crop"></i> Set ROI
                            </button>
                            <button class="btn btn-sm btn-outline-danger remove-camera-btn" data-camera-id="${cameraId}">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    </div>
                    <div id="crowd-count-${cameraId}" class="crowd-count mb-2 normal">Crowd Count: 0</div>
                    <div id="threshold-${cameraId}" class="threshold mb-2">Alert Threshold: ${camera.threshold || '50'}</div>
                    <div id="density-threshold-${cameraId}" class="density-threshold mb-2">Density Threshold: ${camera.density_threshold || '20'}</div>
                    <div id="priority-${cameraId}" class="priority mb-2">Priority: ${camera.priority || '3'}</div>
                    <div id="mode-${cameraId}" class="detection-mode mb-2">Mode: YOLO</div>
                    <div class="performance-metrics">
                        <div id="fps-${cameraId}" class="fps-info mb-2">
                            <div>Target FPS: <span class="target-fps">${camera.fps || '30'}</span></div>
                            <div>Actual FPS: <span class="actual-fps">0.0</span></div>
                        </div>
                        <div id="latency-${cameraId}" class="latency mb-2">Processing Time: <span class="latency-value">0.0</span> ms</div>
                    </div>
                    <div class="mt-2">
                        <button class="btn btn-sm btn-outline-secondary set-threshold-btn" data-camera-id="${cameraId}" data-bs-toggle="modal" data-bs-target="#thresholdModal">
                            <i class="bi bi-sliders"></i> Settings
                        </button>
                    </div>
                </div>
            </div>
        `;
        container.appendChild(card);
    }
    
    // Add event listeners to buttons
    document.querySelectorAll('.set-roi-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const cameraId = this.getAttribute('data-camera-id');
            openRoiModal(cameraId);
        });
    });
    
    document.querySelectorAll('.remove-camera-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const cameraId = this.getAttribute('data-camera-id');
            removeCamera(cameraId);
        });
    });
}

// Add new camera
function addCamera() {
    const cameraId = document.getElementById('camera-id').value;
    const cameraName = document.getElementById('camera-name').value;
    const cameraUrl = document.getElementById('camera-url').value;
    const threshold = parseInt(document.getElementById('camera-threshold').value);
    
    if (!cameraId || !cameraName || !cameraUrl) {
        alert('Please fill all required fields');
        return;
    }
    
    const cameraData = {
        id: cameraId,
        name: cameraName,
        url: cameraUrl,
        threshold: threshold
    };
    
    fetch('/api/add_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(cameraData)
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            // Close modal and reload cameras
            const modal = bootstrap.Modal.getInstance(document.getElementById('addCameraModal'));
            modal.hide();
            loadCameras();
            
            // Clear form
            document.getElementById('add-camera-form').reset();
        } else {
            alert('Error adding camera: ' + (result.message || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error adding camera:', error);
        alert('Error adding camera');
    });
}

// Remove camera
function removeCamera(cameraId) {
    if (!confirm(`Are you sure you want to remove camera ${cameraId}?`)) {
        return;
    }
    
    fetch('/api/remove_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ id: cameraId })
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            loadCameras();
        } else {
            alert('Error removing camera: ' + (result.message || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error removing camera:', error);
        alert('Error removing camera');
    });
}

// Update the ensureRoiCanvasInitialized function
function ensureRoiCanvasInitialized() {
    const canvas = document.getElementById('roi-canvas');
    if (!canvas) {
        console.error('ROI canvas not found');
        return false;
    }

    try {
        // Set default dimensions to match YOLO model input
        canvas.width = 640;
        canvas.height = 640;

        // Get context with willReadFrequently flag
        const ctx = canvas.getContext('2d', {
            willReadFrequently: true,
            alpha: false
        });
        
        if (!ctx) {
            console.error('Failed to get canvas context');
            return false;
        }

        // Store the context
        roiContext = ctx;
        roiCanvas = canvas;

        return true;
    } catch (error) {
        console.error('Error initializing ROI canvas:', error);
        return false;
    }
}

// Update the openRoiModal function's canvas initialization
function openRoiModal(cameraId) {
    if (!ensureRoiCanvasInitialized()) {
        console.error('Failed to initialize ROI canvas');
        return;
    }
    
    currentCameraId = cameraId;
    roiPoints = [];
    
    const loadingIndicator = document.getElementById('canvas-loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
    }
    
    const img = new Image();
    img.crossOrigin = "anonymous";
    
    img.onload = function() {
        if (this.width === 0 || this.height === 0) {
            console.error('Image has zero dimensions');
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            showToast("Invalid image dimensions. Please try again.", "error");
            return;
        }
        
        try {
            // Set canvas dimensions to match image
            roiCanvas.width = this.width;
            roiCanvas.height = this.height;
            
            // Store image reference
            cameraImage = this;
            
            // Use requestAnimationFrame to ensure canvas is ready
            requestAnimationFrame(() => {
                try {
                    if (roiCanvas.width > 0 && roiCanvas.height > 0) {
                        roiContext.drawImage(this, 0, 0);
                        
                        // Verify canvas state with getImageData
                        const imageData = roiContext.getImageData(0, 0, roiCanvas.width, roiCanvas.height);
                        console.log("Canvas ready, image data size:", imageData.data.length);
                        
                        if (loadingIndicator) {
                            loadingIndicator.style.display = 'none';
                        }
                        
                        loadExistingRoi(cameraId);
                        updateRoiPointsDisplay();
                    }
                } catch (error) {
                    console.error("Error drawing on canvas:", error);
                    if (loadingIndicator) {
                        loadingIndicator.style.display = 'none';
                    }
                    showToast("Error processing image. Please try again.", "error");
                }
            });
        } catch (error) {
            console.error('Error processing image:', error);
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            showToast("Error processing image. Please try again.", "error");
        }
    };
    
    // Add timestamp to prevent caching
    const timestamp = Date.now();
    img.src = `/video_feed/${cameraId}?t=${timestamp}&static=true`;
}

// Set up ROI modal buttons
function setupRoiModalButtons() {
    // Rectangle button
    const rectangleBtn = document.getElementById('rectangle-btn');
    if (rectangleBtn) {
        rectangleBtn.onclick = function() {
            createRectangleRoi();
        };
    }
    
    // Circle button
    const circleBtn = document.getElementById('circle-btn');
    if (circleBtn) {
        circleBtn.onclick = function() {
            createCircleRoi();
        };
    }
    
    // Top Half button
    const topHalfBtn = document.getElementById('top-half-btn');
    if (topHalfBtn) {
        topHalfBtn.onclick = function() {
            createTopHalfRoi();
        };
    }
    
    // Bottom Half button
    const bottomHalfBtn = document.getElementById('bottom-half-btn');
    if (bottomHalfBtn) {
        bottomHalfBtn.onclick = function() {
            createBottomHalfRoi();
        };
    }
    
    // Left Half button
    const leftHalfBtn = document.getElementById('left-half-btn');
    if (leftHalfBtn) {
        leftHalfBtn.onclick = function() {
            createLeftHalfRoi();
        };
    }
    
    // Right Half button
    const rightHalfBtn = document.getElementById('right-half-btn');
    if (rightHalfBtn) {
        rightHalfBtn.onclick = function() {
            createRightHalfRoi();
        };
    }
    
    // Full Frame button
    const fullFrameBtn = document.getElementById('full-frame-roi-btn');
    if (fullFrameBtn) {
        fullFrameBtn.onclick = function() {
            createFullFrameRoi();
        };
    }
    
    // Save ROI button
    const saveRoiBtn = document.getElementById('save-roi-btn');
    if (saveRoiBtn) {
        saveRoiBtn.onclick = function() {
            saveRoi();
        };
    }
    
    // Clear ROI button
    const clearRoiBtn = document.getElementById('clear-roi-btn');
    if (clearRoiBtn) {
        clearRoiBtn.onclick = function() {
            clearRoiPoints();
        };
    }
    
    // Zoom buttons
    const zoomInBtn = document.getElementById('zoom-in-btn');
    if (zoomInBtn) {
        zoomInBtn.onclick = function() {
            zoomIn();
        };
    }
    
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    if (zoomOutBtn) {
        zoomOutBtn.onclick = function() {
            zoomOut();
        };
    }
    
    const resetViewBtn = document.getElementById('reset-view-btn');
    if (resetViewBtn) {
        resetViewBtn.onclick = function() {
            resetView();
        };
    }
}

// Create rectangle ROI
function createRectangleRoi() {
    if (!cameraImage) {
        console.error("Cannot create rectangle ROI - no camera image loaded");
        return;
    }
    
    // Clear existing points
    roiPoints = [];
    
    // Get image dimensions
    const width = cameraImage.originalWidth;
    const height = cameraImage.originalHeight;
    
    // Create rectangle with margins
    const margin = 0.1; // 10% margin
    const startX = Math.round(width * margin);
    const startY = Math.round(height * margin);
    const endX = Math.round(width * (1 - margin));
    const endY = Math.round(height * (1 - margin));
    
    // Add points for rectangle
    roiPoints.push([startX, startY]); // Top-left
    roiPoints.push([endX, startY]); // Top-right
    roiPoints.push([endX, endY]); // Bottom-right
    roiPoints.push([startX, endY]); // Bottom-left
    
    // Redraw canvas
    redrawRoiCanvas();
    
    console.log("Created rectangle ROI");
}

// Create full frame ROI
function createFullFrameRoi() {
    if (!cameraImage) {
        console.error("Cannot create full frame ROI - no camera image loaded");
        return;
    }
    
    // Clear existing points
    roiPoints = [];
    
    // Get image dimensions
    const width = cameraImage.originalWidth;
    const height = cameraImage.originalHeight;
    
    // Add points for the full frame
    roiPoints.push([0, 0]); // Top-left
    roiPoints.push([width, 0]); // Top-right
    roiPoints.push([width, height]); // Bottom-right
    roiPoints.push([0, height]); // Bottom-left
    
    // Redraw canvas
    redrawRoiCanvas();
    
    console.log("Created full frame ROI");
}

// Initialize ROI template handling
function initializeRoiTemplateHandling() {
    // Add event listeners for template save and load buttons
    const saveTemplateBtn = document.getElementById('save-roi-as-btn');
    if (saveTemplateBtn) {
        saveTemplateBtn.onclick = saveRoiTemplate;
    }
    
    const loadTemplateBtn = document.getElementById('load-roi-template-btn');
    if (loadTemplateBtn) {
        loadTemplateBtn.onclick = loadRoiTemplate;
    }
    
    // Load templates from local storage
    loadRoiTemplatesFromStorage();
}

// Save ROI template
function saveRoiTemplate() {
    if (!roiPoints || roiPoints.length < 3) {
        alert('Please select at least 3 points to save a template.');
        return;
    }
    
    const templateName = prompt('Enter a name for this ROI template:');
    if (!templateName) return;
    
    // Initialize roiTemplates if not exists
    if (!roiTemplates) {
        roiTemplates = {};
    }
    
    // Save template
    roiTemplates[templateName] = [...roiPoints];
    
    // Store in localStorage
    try {
        localStorage.setItem('roiTemplates', JSON.stringify(roiTemplates));
        alert(`Template "${templateName}" saved successfully.`);
    } catch (e) {
        console.error('Error saving ROI template:', e);
        alert('Error saving template. Local storage may be full or disabled.');
    }
}

// Load ROI template
function loadRoiTemplate() {
    // Initialize roiTemplates if not exists
    if (!roiTemplates) {
        loadRoiTemplatesFromStorage();
    }
    
    // Get template names
    const templateNames = Object.keys(roiTemplates || {});
    if (templateNames.length === 0) {
        alert('No saved templates found.');
        return;
    }
    
    // Create selection options
    const template = prompt('Enter template name to load:\n\nAvailable templates: ' + templateNames.join(', '));
    if (!template || !roiTemplates[template]) {
        if (template) alert('Template not found.');
        return;
    }
    
    if (!cameraImage) {
        alert('Cannot load template - no camera image is loaded');
        return;
    }
    
    // Load template points with scaling
    const templatePoints = roiTemplates[template];
    
    // Clear current points
    roiPoints = [];
    
    // Add each point with proper scaling if needed
    for (const point of templatePoints) {
        // Coordinates in template could be from a different sized image
        // We'll just use them directly since our rendering already handles scaling
        roiPoints.push([...point]); // Clone point to avoid modifying the template
    }
    
    console.log(`Loaded template '${template}' with ${roiPoints.length} points`);
    
    // Redraw canvas
    redrawRoiCanvas();
}

// Load ROI templates from localStorage
function loadRoiTemplatesFromStorage() {
    try {
        const savedTemplates = localStorage.getItem('roiTemplates');
        if (savedTemplates) {
            roiTemplates = JSON.parse(savedTemplates);
            console.log("ROI templates loaded:", Object.keys(roiTemplates));
        } else {
            roiTemplates = {};
        }
    } catch (e) {
        console.error('Error loading ROI templates:', e);
        roiTemplates = {};
    }
}

// Add ROI point on canvas click
function addRoiPoint(event) {
    if (!ensureRoiCanvasInitialized()) {
        return;
    }
    
    if (!cameraImage) {
        console.error("Cannot add ROI point - missing camera image");
        showToast("Cannot add point - camera image not loaded", "error");
        return;
    }
    
    if (roiCanvas.width === 0 || roiCanvas.height === 0 || !cameraImage.originalWidth || !cameraImage.originalHeight) {
        console.error("Cannot add ROI point - canvas or image has zero dimensions");
        showToast("Cannot add point - image/canvas has zero dimensions", "error");
        return;
    }
    
    // Get click coordinates relative to canvas
    const rect = roiCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Calculate scale factor between displayed size and actual size
    const scaleX = roiCanvas.width / rect.width;
    const scaleY = roiCanvas.height / rect.height;
    
    // Adjust coordinates based on scaling
    const canvasX = x * scaleX;
    const canvasY = y * scaleY;
    
    console.log(`Canvas click at: ${canvasX}, ${canvasY}`);
    
    // Check if we're clicking near an existing point to remove it
    const pointRadius = 10; // Detection radius in pixels
    let clickedOnPoint = false;
    
    for (let i = roiPoints.length - 1; i >= 0; i--) {
        const point = roiPoints[i];
        
        // Convert original coordinates to canvas coordinates
        const pointCanvasX = (point[0] / cameraImage.originalWidth) * roiCanvas.width;
        const pointCanvasY = (point[1] / cameraImage.originalHeight) * roiCanvas.height;
        
        // Calculate distance
        const distance = Math.sqrt(
            Math.pow(canvasX - pointCanvasX, 2) + 
            Math.pow(canvasY - pointCanvasY, 2)
        );
        
        if (distance <= pointRadius) {
            // Remove this point
            roiPoints.splice(i, 1);
            clickedOnPoint = true;
            console.log(`Removed point at index ${i}`);
            break;
        }
    }
    
    // If we didn't click on a point, add a new one
    if (!clickedOnPoint) {
        // Convert canvas coordinates to original image coordinates
        const imgX = Math.round((canvasX / roiCanvas.width) * cameraImage.originalWidth);
        const imgY = Math.round((canvasY / roiCanvas.height) * cameraImage.originalHeight);
        
        // Add the point using original image coordinates
        roiPoints.push([imgX, imgY]);
        console.log(`Added ROI point at ${imgX}, ${imgY}`);
    }
    
    // Redraw the canvas to show the changes
    redrawRoiCanvas();
    updateRoiPointsDisplay();
}

// Redraw ROI canvas
function redrawRoiCanvas() {
    if (!ensureRoiCanvasInitialized()) {
        return;
    }
    
    if (!cameraImage) {
        console.error("Cannot redraw ROI canvas - missing camera image");
        return;
    }
    
    if (roiCanvas.width === 0 || roiCanvas.height === 0 || !cameraImage.originalWidth || !cameraImage.originalHeight) {
        console.error("Cannot redraw ROI canvas - canvas or image has zero dimensions");
        roiContext.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        roiContext.fillStyle = "#ffeeee";
        roiContext.fillRect(0, 0, roiCanvas.width, roiCanvas.height);
        roiContext.font = "16px Arial";
        roiContext.fillStyle = "red";
        roiContext.textAlign = "center";
        roiContext.fillText("Error: Image/canvas has zero dimensions", roiCanvas.width/2, roiCanvas.height/2);
        return;
    }
    
    try {
        // Clear canvas
        roiContext.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        
        // Draw image
        roiContext.drawImage(cameraImage, 0, 0, roiCanvas.width, roiCanvas.height);
        
        // Display canvas info for debugging
        roiContext.fillStyle = "rgba(0, 0, 0, 0.5)";
        roiContext.fillRect(0, 0, 200, 60);
        roiContext.font = "10px monospace";
        roiContext.fillStyle = "white";
        roiContext.textAlign = "left";
        roiContext.fillText(`Canvas: ${roiCanvas.width}×${roiCanvas.height}`, 5, 15);
        roiContext.fillText(`Image: ${cameraImage.originalWidth}×${cameraImage.originalHeight}`, 5, 30);
        roiContext.fillText(`Points: ${roiPoints.length}`, 5, 45);
        
        // Draw points and lines only if we have points
        if (roiPoints.length > 0) {
            // Convert points to canvas coordinates
            const canvasPoints = roiPoints.map(point => {
                return [
                    (point[0] / cameraImage.originalWidth) * roiCanvas.width,
                    (point[1] / cameraImage.originalHeight) * roiCanvas.height
                ];
            });
            
            // Draw polygon if we have at least 3 points
            if (canvasPoints.length >= 3) {
                // Draw filled polygon
                roiContext.beginPath();
                roiContext.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
                
                for (let i = 1; i < canvasPoints.length; i++) {
                    roiContext.lineTo(canvasPoints[i][0], canvasPoints[i][1]);
                }
                
                // Close the polygon
                roiContext.closePath();
                
                // Fill with semi-transparent color
                roiContext.fillStyle = "rgba(255, 0, 0, 0.2)";
                roiContext.fill();
                
                // Draw outline
                roiContext.strokeStyle = "rgba(255, 0, 0, 0.8)";
                roiContext.lineWidth = 2;
                roiContext.stroke();
            } else {
                // Just draw lines if we have less than 3 points
                roiContext.beginPath();
                roiContext.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
                
                for (let i = 1; i < canvasPoints.length; i++) {
                    roiContext.lineTo(canvasPoints[i][0], canvasPoints[i][1]);
                }
                
                roiContext.strokeStyle = "rgba(255, 0, 0, 0.8)";
                roiContext.lineWidth = 2;
                roiContext.stroke();
            }
            
            // Draw the points
            canvasPoints.forEach((point, i) => {
                // Outer circle
                roiContext.beginPath();
                roiContext.arc(point[0], point[1], 8, 0, 2 * Math.PI);
                roiContext.fillStyle = "rgba(255, 255, 255, 0.8)";
                roiContext.fill();
                
                // Inner circle
                roiContext.beginPath();
                roiContext.arc(point[0], point[1], 5, 0, 2 * Math.PI);
                roiContext.fillStyle = "rgba(255, 0, 0, 0.9)";
                roiContext.fill();
                
                // Point number
                roiContext.font = "10px Arial";
                roiContext.fillStyle = "white";
                roiContext.textAlign = "center";
                roiContext.fillText(i + 1, point[0], point[1] + 3);
            });
        }
    } catch (error) {
        console.error("Error redrawing ROI canvas:", error);
        
        // Show error message on canvas
        roiContext.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        roiContext.fillStyle = "#ffeeee";
        roiContext.fillRect(0, 0, roiCanvas.width, roiCanvas.height);
        roiContext.font = "16px Arial";
        roiContext.fillStyle = "red";
        roiContext.textAlign = "center";
        roiContext.fillText("Error drawing ROI canvas", roiCanvas.width/2, roiCanvas.height/2 - 20);
        roiContext.fillText(error.toString(), roiCanvas.width/2, roiCanvas.height/2 + 20);
    }
}

// Update ROI points display with improved formatting
function updateRoiPointsDisplay() {
    const pointsEl = document.getElementById('roi-points-list');
    if (!pointsEl) return;
    
    if (roiPoints.length === 0) {
        pointsEl.innerHTML = '<span class="text-muted">No points selected</span>';
        return;
    }
    
    let html = '';
    roiPoints.forEach((point, index) => {
        html += `
            <div class="point-item mb-1">
                <span class="badge bg-secondary me-1">Point ${index + 1}</span>
                <small>X: ${point[0]}, Y: ${point[1]}</small>
            </div>
        `;
    });
    
    // Add validation message
    if (roiPoints.length < 3) {
        html += '<div class="alert alert-warning mt-2 py-1 px-2 small">At least 3 points needed for a valid region</div>';
    } else {
        html += '<div class="alert alert-success mt-2 py-1 px-2 small">Valid region selected</div>';
    }
    
    pointsEl.innerHTML = html;
}

// Helper function to show a toast message
function showToast(message, type = 'info') {
    // Check if we have a toast container, if not create one
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastEl = document.createElement('div');
    toastEl.className = `toast ${type === 'error' ? 'bg-danger text-white' : 'bg-dark text-white'}`;
    toastEl.id = toastId;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    toastEl.innerHTML = `
        <div class="toast-header">
            <strong class="me-auto">${type === 'error' ? 'Error' : 'Message'}</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    // Add to container
    toastContainer.appendChild(toastEl);
    
    // Initialize and show the toast
    const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
    toast.show();
}

// Clear ROI points
function clearRoiPoints() {
    roiPoints = [];
    console.log("Cleared all ROI points");
    redrawRoiCanvas();
}

// Add ROI validation function
function validateRoiPoints(points, frameWidth, frameHeight) {
    return points.every(point => {
        const [x, y] = point;
        return x >= 0 && x < frameWidth && y >= 0 && y < frameHeight;
    });
}

// Update saveRoi function with improved validation
function saveRoi() {
    if (!currentCameraId || roiPoints.length < 3) {
        showToast("Please select at least 3 points to define the ROI", "error");
        return;
    }

    // Show loading state
    const saveBtn = document.getElementById('save-roi-btn');
    if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';
    }

    // Get camera info first
    fetch(`/api/cameras/${currentCameraId}`)
        .then(response => response.json())
        .then(cameraInfo => {
            const frameWidth = cameraInfo.width || 640;
            const frameHeight = cameraInfo.height || 480;

            // Format and validate points
            const formattedPoints = roiPoints.map(point => [
                Math.round(point[0]),
                Math.round(point[1])
            ]);

            if (!validateRoiPoints(formattedPoints, frameWidth, frameHeight)) {
                showToast("ROI points must be within frame bounds", "error");
                return;
            }

            // Proceed with saving ROI
            return fetch('/api/set_roi', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    id: currentCameraId,
                    roi: formattedPoints
                })
            });
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('roiModal'));
                modal.hide();
                
                // Reset variables
                currentCameraId = null;
                roiPoints = [];
                
                // Show success message
                showToast("Region of Interest saved successfully", "success");
                
                // Refresh data
                refreshData();
            } else {
                showToast(result.message || "Error saving ROI", "error");
            }
        })
        .catch(error => {
            console.error('Error saving ROI:', error);
            showToast("Error saving ROI", "error");
        })
        .finally(() => {
            // Restore button state
            if (saveBtn) {
                saveBtn.disabled = false;
                saveBtn.innerHTML = 'Save ROI';
            }
        });
}

// Load existing ROI for a camera
function loadExistingRoi(cameraId) {
    fetch('/api/cameras')
        .then(response => response.json())
        .then(cameras => {
            const camera = cameras[cameraId];
            if (camera && camera.roi) {
                roiPoints = camera.roi;
                redrawRoiCanvas();
                updateRoiPointsDisplay();
                updateRoiStatistics();
            }
        })
        .catch(error => {
            console.error('Error loading existing ROI:', error);
            showToast("Error loading existing ROI", "error");
        });
}

// Refresh data
function refreshData() {
    // Update crowd data
    fetch('/api/crowd_data')
        .then(response => response.json())
        .then(data => {
            updateCrowdData(data);
            updateCrowdChart(data);
        })
        .catch(error => {
            console.error('Error fetching crowd data:', error);
        });
    
    // Update alerts
    fetch('/api/alerts')
        .then(response => response.json())
        .then(alerts => {
            updateAlerts(alerts);
        })
        .catch(error => {
            console.error('Error fetching alerts:', error);
        });
    
    // Update performance data
    fetch('/api/performance')
        .then(response => response.json())
        .then(data => {
            updatePerformanceChart(data);
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
        });
}

// Update crowd data in the UI
function updateCrowdData(data) {
    for (const [cameraId, cameraData] of Object.entries(data)) {
        const countElement = document.getElementById(`crowd-count-${cameraId}`);
        const modelElement = document.getElementById(`model-info-${cameraId}`);
        const densityElement = document.getElementById(`density-threshold-${cameraId}`);
        const modeElement = document.getElementById(`mode-${cameraId}`);
        if (countElement) {
            const count = cameraData.current_count;
            const thresholdElement = document.getElementById(`threshold-${cameraId}`);
            const threshold = thresholdElement ? parseInt(thresholdElement.textContent) : 0;
            countElement.textContent = `Crowd Count: ${count}`;

            // Update model info and mode
            if (modeElement) {
                if (typeof cameraData.detection_mode === 'object' && cameraData.detection_mode !== null) {
                    modeElement.textContent = cameraData.detection_mode.mode || '';
                    if (modelElement && cameraData.detection_mode.model) {
                        modelElement.textContent = `Model: ${cameraData.detection_mode.model}`;
                    }
                    if (densityElement && cameraData.detection_mode.threshold) {
                        densityElement.textContent = `Density Threshold: ${cameraData.detection_mode.threshold}`;
                    }
                } else {
                    modeElement.textContent = cameraData.detection_mode || '';
                    if (modelElement && cameraData.model_info) {
                        modelElement.textContent = `Model: ${cameraData.model_info}`;
                    }
                    if (densityElement && cameraData.density_threshold) {
                        densityElement.textContent = `Density Threshold: ${cameraData.density_threshold}`;
                    }
                }
            }

            // Update color based on threshold
            if (threshold > 0) {
                if (count >= threshold) {
                    countElement.className = 'crowd-count mb-2 danger';
                } else if (count >= threshold * 0.8) {
                    countElement.className = 'crowd-count mb-2 warning';
                } else {
                    countElement.className = 'crowd-count mb-2 normal';
                }
            }
        }
    }
}

// Update crowd trends chart
function updateCrowdChart(data) {
    if (!crowdTrendsChart) return;
    
    // Reset datasets
    crowdTrendsChart.data.datasets = [];
    
    // Add dataset for each camera
    let maxDataPoints = 0;
    for (const [cameraId, cameraData] of Object.entries(data)) {
        if (!cameraData.history || cameraData.history.length === 0) continue;
        
        // Create dataset for this camera
        const dataset = {
            label: `Camera ${cameraId}`,
            data: cameraData.history.map(h => h.count),
            borderColor: getRandomColor(),
            tension: 0.1
        };
        
        crowdTrendsChart.data.datasets.push(dataset);
        
        // Update max data points
        maxDataPoints = Math.max(maxDataPoints, cameraData.history.length);
    }
    
    // Update labels (times)
    if (maxDataPoints > 0 && data[Object.keys(data)[0]].history) {
        crowdTrendsChart.data.labels = data[Object.keys(data)[0]].history.map(h => h.datetime);
    }
    
    // Update chart
    crowdTrendsChart.update();
}

// Update performance chart
function updatePerformanceChart(data) {
    if (!performanceChart) return;
    
    console.log('Performance data received:', data);
    
    // Update datasets
    if (data.fps && data.fps.length > 0) {
        console.log('FPS data:', data.fps);
        performanceChart.data.datasets[0].data = data.fps.map(item => item.value);
    }
    
    if (data.latency && data.latency.length > 0) {
        console.log('Latency data:', data.latency);
        performanceChart.data.datasets[1].data = data.latency.map(item => item.value * 1000); // Convert to ms
    }
    
    // Update labels (times) with correct timestamp format
    if (data.fps && data.fps.length > 0) {
        performanceChart.data.labels = data.fps.map(item => {
            // Use the datetime string directly from the server if available
            if (item.datetime) {
                return item.datetime;
            }
            
            // Or properly format the timestamp
            const time = new Date(item.timestamp * 1000); // Ensure timestamp is treated as seconds
            return time.toLocaleTimeString();
        });
    }
    
    // Update chart
    performanceChart.update();
}

// Update alerts in the UI
function updateAlerts(alerts) {
    const container = document.getElementById('alerts-container');
    if (!container) return;
    
    // Check if alerts is empty
    if (alerts.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No alerts yet.</div>';
        return;
    }
    
    // Sort alerts by time (newest first)
    alerts.sort((a, b) => b.timestamp - a.timestamp);
    
    // Clear container
    container.innerHTML = '';
    
    // Create alert cards
    for (const alert of alerts) {
        const card = document.createElement('div');
        card.className = 'alert-card p-3 mb-3';
        
        // Determine severity class
        let severityClass = '';
        if (alert.severity === 'high') {
            severityClass = 'danger';
        } else if (alert.severity === 'medium') {
            severityClass = 'warning';
        } else {
            severityClass = 'normal';
        }
        
        // Format actions as list
        let actionsHtml = '';
        if (alert.actions && alert.actions.length > 0) {
            actionsHtml = '<ul class="mb-0 mt-2">';
            for (const action of alert.actions) {
                actionsHtml += `<li>${action}</li>`;
            }
            actionsHtml += '</ul>';
        }
        
        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <h5 class="mb-1 ${severityClass}">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i>
                        ${alert.severity.toUpperCase()} Alert - Camera ${alert.camera_id}
                    </h5>
                    <div class="text-muted small">${alert.datetime}</div>
                </div>
                <div class="badge bg-${severityClass === 'normal' ? 'success' : (severityClass === 'warning' ? 'warning' : 'danger')} p-2">
                    Count: ${alert.crowd_count} / Threshold: ${alert.threshold}
                </div>
            </div>
            <hr class="my-2">
            <div>
                <strong>Suggested Actions:</strong>
                ${actionsHtml}
            </div>
        `;
        
        container.appendChild(card);
    }

    // Play alert sound
    playAlertSound();
}

// Generate random color for charts
function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

// Create circle ROI
function createCircleRoi() {
    if (!cameraImage) {
        console.error("Cannot create circle ROI - no camera image loaded");
        return;
    }
    
    // Clear existing points
    roiPoints = [];
    
    // Get image dimensions
    const width = cameraImage.originalWidth;
    const height = cameraImage.originalHeight;
    const centerX = Math.round(width / 2);
    const centerY = Math.round(height / 2);
    
    // Calculate radius (40% of the smaller dimension)
    const radius = Math.round(Math.min(width, height) * 0.4);
    
    // Create a circle approximation with points
    const numPoints = 16;  // Number of points to use for the circle
    for (let i = 0; i < numPoints; i++) {
        const angle = (i / numPoints) * 2 * Math.PI;
        const x = Math.round(centerX + radius * Math.cos(angle));
        const y = Math.round(centerY + radius * Math.sin(angle));
        roiPoints.push([x, y]);
    }
    
    // Redraw canvas
    redrawRoiCanvas();
    updateRoiPointsDisplay();
    
    console.log("Created circle ROI");
}

// Create top half ROI
function createTopHalfRoi() {
    if (!cameraImage) {
        console.error("Cannot create top half ROI - no camera image loaded");
        return;
    }
    
    // Clear existing points
    roiPoints = [];
    
    // Get image dimensions
    const width = cameraImage.originalWidth;
    const height = cameraImage.originalHeight;
    const halfHeight = Math.round(height / 2);
    
    // Add points for the top half
    roiPoints.push([0, 0]);               // Top-left
    roiPoints.push([width, 0]);           // Top-right
    roiPoints.push([width, halfHeight]);  // Middle-right
    roiPoints.push([0, halfHeight]);      // Middle-left
    
    // Redraw canvas
    redrawRoiCanvas();
    updateRoiPointsDisplay();
    
    console.log("Created top half ROI");
}

// Create bottom half ROI
function createBottomHalfRoi() {
    if (!cameraImage) {
        console.error("Cannot create bottom half ROI - no camera image loaded");
        return;
    }
    
    // Clear existing points
    roiPoints = [];
    
    // Get image dimensions
    const width = cameraImage.originalWidth;
    const height = cameraImage.originalHeight;
    const halfHeight = Math.round(height / 2);
    
    // Add points for the bottom half
    roiPoints.push([0, halfHeight]);      // Middle-left
    roiPoints.push([width, halfHeight]);  // Middle-right
    roiPoints.push([width, height]);      // Bottom-right
    roiPoints.push([0, height]);          // Bottom-left
    
    // Redraw canvas
    redrawRoiCanvas();
    updateRoiPointsDisplay();
    
    console.log("Created bottom half ROI");
}

// Create left half ROI
function createLeftHalfRoi() {
    if (!cameraImage) {
        console.error("Cannot create left half ROI - no camera image loaded");
        return;
    }
    
    // Clear existing points
    roiPoints = [];
    
    // Get image dimensions
    const width = cameraImage.originalWidth;
    const height = cameraImage.originalHeight;
    const halfWidth = Math.round(width / 2);
    
    // Add points for the left half
    roiPoints.push([0, 0]);               // Top-left
    roiPoints.push([halfWidth, 0]);       // Top-middle
    roiPoints.push([halfWidth, height]);  // Bottom-middle
    roiPoints.push([0, height]);          // Bottom-left
    
    // Redraw canvas
    redrawRoiCanvas();
    updateRoiPointsDisplay();
    
    console.log("Created left half ROI");
}

// Create right half ROI
function createRightHalfRoi() {
    if (!cameraImage) {
        console.error("Cannot create right half ROI - no camera image loaded");
        return;
    }
    
    // Clear existing points
    roiPoints = [];
    
    // Get image dimensions
    const width = cameraImage.originalWidth;
    const height = cameraImage.originalHeight;
    const halfWidth = Math.round(width / 2);
    
    // Add points for the right half
    roiPoints.push([halfWidth, 0]);       // Top-middle
    roiPoints.push([width, 0]);           // Top-right
    roiPoints.push([width, height]);      // Bottom-right
    roiPoints.push([halfWidth, height]);  // Bottom-middle
    
    // Redraw canvas
    redrawRoiCanvas();
    updateRoiPointsDisplay();
    
    console.log("Created right half ROI");
}

// Zoom and view functions
function zoomIn() {
    if (!roiCanvas || !cameraImage) return;
    
    // Increase scale by 20%
    const currentScale = roiCanvas.width / cameraImage.originalWidth;
    const newScale = currentScale * 1.2;
    
    // Update canvas size
    roiCanvas.width = cameraImage.originalWidth * newScale;
    roiCanvas.height = cameraImage.originalHeight * newScale;
    
    // Redraw
    redrawRoiCanvas();
}

function zoomOut() {
    if (!roiCanvas || !cameraImage) return;
    
    // Decrease scale by 20%
    const currentScale = roiCanvas.width / cameraImage.originalWidth;
    const newScale = currentScale / 1.2;
    
    // Update canvas size
    roiCanvas.width = cameraImage.originalWidth * newScale;
    roiCanvas.height = cameraImage.originalHeight * newScale;
    
    // Redraw
    redrawRoiCanvas();
}

function resetView() {
    if (!roiCanvas || !cameraImage) return;
    
    // Reset to original size
    roiCanvas.width = cameraImage.originalWidth;
    roiCanvas.height = cameraImage.originalHeight;
    
    // Redraw
    redrawRoiCanvas();
}

function resetCameras() {
    if (confirm('Are you sure you want to reset all cameras? This will clear all camera configurations and data.')) {
        fetch('/api/reset_cameras', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(data.message);
                // Reload the page to reflect changes
                window.location.reload();
            } else {
                alert(data.message || 'Error resetting cameras');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error resetting cameras: ' + error.message);
        });
    }
}

function playAlertSound() {
    if (!userInteracted) return; // Don't play if user hasn't interacted
    const alertSound = document.getElementById('alert-sound');
    if (alertSound) {
        alertSound.play().catch(error => {
            console.error('Error playing alert sound:', error);
        });
    }
}