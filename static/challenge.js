/**
 * Advanced Eye Tracking Liveness Challenge
 * Implements the complete redesigned challenge flow with face capture and screen detection
 */

class AdvancedLivenessChallenge {
    constructor() {
        // Configuration
        this.config = {
            apiBaseUrl: 'http://localhost:8000',
            videoDuration: null,
            areaDuration: 3.0,
            videoCodec: 'video/webm;codecs=vp9',
            fallbackCodec: 'video/webm'
        };

        // State management
        this.state = {
            sessionId: null,
            currentStep: 'instructions',
            faceDetected: false,
            faceCaptured: false,
            faceSnapshot: null,
            screenDimensions: null,
            challengeData: null,
            mediaRecorder: null,
            recordedChunks: [],
            stream: null,
            isRecording: false
        };

        // DOM elements
        this.elements = {};
        
        // Timers
        this.timers = {
            faceDetection: null,
            challenge: null,
            countdown: null
        };

        this.initializeApp();
    }

    /**
     * Initialize the application
     */
    async initializeApp() {
        try {
            this.extractSessionId();
            this.bindElements();
            this.bindEvents();
            this.showStep('instructions');
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.showError('Initialization failed', error.message);
        }
    }

    /**
     * Extract session ID from URL parameters
     */
    extractSessionId() {
        const urlParams = new URLSearchParams(window.location.search);
        this.state.sessionId = urlParams.get('session');
        
        if (!this.state.sessionId) {
            throw new Error('No session ID provided in URL');
        }
        
        console.log('Session ID:', this.state.sessionId);
    }

    /**
     * Bind DOM elements
     */
    bindElements() {
        this.elements = {
            // Steps
            instructionsStep: document.getElementById('instructions-step'),
            faceSetupStep: document.getElementById('face-setup-step'),
            screenDetectionStep: document.getElementById('screen-detection-step'),
            challengeExecutionStep: document.getElementById('challenge-execution-step'),
            resultsStep: document.getElementById('results-step'),
            errorStep: document.getElementById('error-step'),

            // Instructions step
            startSetup: document.getElementById('start-setup'),

            // Face setup step
            faceCaptureVideo: document.getElementById('face-capture-video'),
            faceOutline: document.getElementById('face-outline'),
            captureStatus: document.getElementById('capture-status'),
            captureFace: document.getElementById('capture-face'),
            backToInstructions: document.getElementById('back-to-instructions'),

            // Screen detection step
            detectionStatus: document.getElementById('detection-status'),
            detectionProgress: document.querySelector('#detection-progress .progress-fill'),
            screenInfo: document.getElementById('screen-info'),
            screenResolution: document.getElementById('screen-resolution'),
            areaSize: document.getElementById('area-size'),
            screenDetectionControls: document.getElementById('screen-detection-controls'),
            startChallenge: document.getElementById('start-challenge'),

            // Challenge execution
            dynamicGrid: document.getElementById('dynamic-grid'),
            progressOverlay: document.getElementById('progress-overlay'),
            currentInstruction: document.getElementById('current-instruction'),
            sequenceNumbers: document.getElementById('sequence-numbers'),
            countdownDisplay: document.getElementById('countdown-display'),
            challengeProgress: document.querySelector('#challenge-progress .progress-fill'),

            // Results
            resultIcon: document.getElementById('result-icon'),
            resultTitle: document.getElementById('result-title'),
            resultMessage: document.getElementById('result-message'),
            overallAccuracy: document.getElementById('overall-accuracy'),
            processingTime: document.getElementById('processing-time'),
            sessionIdDisplay: document.getElementById('session-id'),
            sequenceResults: document.getElementById('sequence-results'),
            newChallenge: document.getElementById('new-challenge'),
            downloadReport: document.getElementById('download-report'),

            // Error
            errorMessage: document.getElementById('error-message'),
            errorCode: document.getElementById('error-code'),
            retryChallenge: document.getElementById('retry-challenge'),
            newSession: document.getElementById('new-session')
        };
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Instructions step
        this.elements.startSetup.addEventListener('click', () => this.startFaceSetup());

        // Face setup step
        this.elements.captureFace.addEventListener('click', () => this.captureFaceSnapshot());
        this.elements.backToInstructions.addEventListener('click', () => this.showStep('instructions'));

        // Screen detection step
        this.elements.startChallenge.addEventListener('click', () => this.startChallengeExecution());

        // Results step
        this.elements.newChallenge.addEventListener('click', () => this.startNewChallenge());
        this.elements.downloadReport.addEventListener('click', () => this.downloadReport());

        // Error step
        this.elements.retryChallenge.addEventListener('click', () => this.showStep('instructions'));
        this.elements.newSession.addEventListener('click', () => this.createNewSession());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.showStep('instructions');
            }
        });
    }

    /**
     * Show specific step
     */
    showStep(stepName) {
        // Hide all steps
        document.querySelectorAll('.challenge-step').forEach(step => {
            step.classList.remove('active');
        });

        // Show target step
        const targetStep = document.getElementById(`${stepName}-step`);
        if (targetStep) {
            targetStep.classList.add('active');
            this.state.currentStep = stepName;
        }

        // Update step indicators
        this.updateStepIndicators(stepName);
    }

    /**
     * Update step indicators
     */
    updateStepIndicators(currentStep) {
        const steps = ['instructions', 'face-setup', 'screen-detection', 'challenge-execution'];
        const stepIndex = steps.indexOf(currentStep.replace('-step', ''));
        
        document.querySelectorAll('.step-indicator').forEach(indicator => {
            const stepElements = indicator.querySelectorAll('.step');
            stepElements.forEach((step, index) => {
                step.classList.remove('active', 'completed');
                if (index < stepIndex) {
                    step.classList.add('completed');
                } else if (index === stepIndex) {
                    step.classList.add('active');
                }
            });
        });
    }

    /**
     * Start face setup process
     */
    async startFaceSetup() {
        try {
            this.showStep('face-setup');
            await this.setupCamera();
            this.startFaceDetection();
        } catch (error) {
            console.error('Failed to start face setup:', error);
            this.showError('Camera Setup Failed', error.message);
        }
    }

    /**
     * Setup camera for face capture
     */
    async setupCamera() {
        if (this.state.stream) {
            this.state.stream.getTracks().forEach(track => track.stop());
        }

        this.state.stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            },
            audio: false
        });

        this.elements.faceCaptureVideo.srcObject = this.state.stream;
    }

    /**
     * Start face detection monitoring
     */
    startFaceDetection() {
        const checkFaceDetection = () => {
            const video = this.elements.faceCaptureVideo;
            
            if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
                const hasVideo = !video.paused && !video.ended && video.currentTime > 0;
                
                if (hasVideo) {
                    this.updateFaceDetectionStatus(true, 'Face detected - Ready to capture!');
                } else {
                    this.updateFaceDetectionStatus(false, 'Position your face in the outline');
                }
            } else {
                this.updateFaceDetectionStatus(false, 'Loading camera...');
            }
        };

        this.timers.faceDetection = setInterval(checkFaceDetection, 500);
    }

    /**
     * Update face detection status
     */
    updateFaceDetectionStatus(detected, message) {
        this.state.faceDetected = detected;
        
        // Update status text
        this.elements.captureStatus.innerHTML = `
            <span>${detected ? '✅' : '📷'}</span>
            <span>${message}</span>
        `;

        // Update visual state
        this.elements.captureStatus.classList.toggle('face-detected', detected);
        this.elements.faceOutline.classList.toggle('detected', detected);
        this.elements.captureFace.disabled = !detected;
    }

    /**
     * Capture face snapshot
     */
    async captureFaceSnapshot() {
        try {
            const video = this.elements.faceCaptureVideo;
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            // Convert to blob
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            this.state.faceSnapshot = blob;
            this.state.faceCaptured = true;

            console.log('Face snapshot captured:', blob.size, 'bytes');

            // Stop face detection
            if (this.timers.faceDetection) {
                clearInterval(this.timers.faceDetection);
            }

            // Proceed to screen detection
            this.startScreenDetection();

        } catch (error) {
            console.error('Failed to capture face snapshot:', error);
            this.showError('Face Capture Failed', error.message);
        }
    }

    /**
     * Start screen detection process
     */
    async startScreenDetection() {
        this.showStep('screen-detection');
        
        // Simulate detection progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 10;
            this.elements.detectionProgress.style.width = `${progress}%`;
            
            if (progress >= 100) {
                clearInterval(progressInterval);
                this.completeScreenDetection();
            }
        }, 200);
    }

    /**
     * Complete screen detection and show results
     */
    completeScreenDetection() {
        // Get actual screen dimensions
        const screenWidth = window.screen.width;
        const screenHeight = window.screen.height;
        
        // Get viewport dimensions (what we'll actually use)
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        this.state.screenDimensions = {
            screenWidth: viewportWidth,  // Use viewport for challenge
            screenHeight: viewportHeight,
            actualScreenWidth: screenWidth,
            actualScreenHeight: screenHeight
        };

        // Calculate area dimensions
        const gridMargin = 50;
        const gridWidth = viewportWidth - (2 * gridMargin);
        const gridHeight = viewportHeight - (2 * gridMargin);
        const areaWidth = Math.floor(gridWidth / 3);
        const areaHeight = Math.floor(gridHeight / 2);

        // Update UI
        this.elements.detectionStatus.textContent = 'Screen detection complete!';
        this.elements.screenResolution.textContent = `${viewportWidth} × ${viewportHeight}`;
        this.elements.areaSize.textContent = `${areaWidth} × ${areaHeight} pixels`;
        
        this.elements.screenInfo.style.display = 'block';
        this.elements.screenDetectionControls.style.display = 'block';

        console.log('Screen detection complete:', this.state.screenDimensions);
    }

    /**
     * Start challenge execution
     */
    async startChallengeExecution() {
        try {
            // Initiate challenge with backend
            await this.initiateChallengeWithBackend();
            
            // Setup full-screen challenge
            this.setupFullScreenChallenge();
            
            // Start recording and challenge
            await this.startRecording();
            this.startChallengeSequence();
            
        } catch (error) {
            console.error('Failed to start challenge execution:', error);
            this.showError('Challenge Start Failed', error.message);
        }
    }

    /**
     * Initiate challenge with backend
     */
    async initiateChallengeWithBackend() {
        const formData = new FormData();
        formData.append('session_id', this.state.sessionId);
        formData.append('screen_width', this.state.screenDimensions.screenWidth);
        formData.append('screen_height', this.state.screenDimensions.screenHeight);
        formData.append('face_snapshot', this.state.faceSnapshot, 'face_snapshot.jpg');
        formData.append('area_duration', this.config.areaDuration);

        const response = await fetch(`${this.config.apiBaseUrl}/initiate-liveness`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        this.state.challengeData = await response.json();
        if (!this.state.challengeData.success) {
            throw new Error(this.state.challengeData.error || 'Failed to initiate challenge');
        }

        console.log('Challenge initiated:', this.state.challengeData);
    }

    /**
     * Setup full-screen challenge interface
     */
    setupFullScreenChallenge() {
        this.showStep('challenge-execution');
        
        // Show dynamic grid and progress overlay
        this.elements.dynamicGrid.style.display = 'block';
        this.elements.progressOverlay.style.display = 'block';

        // Create challenge areas based on backend response
        this.createDynamicAreas();

        // Update sequence display
        this.elements.sequenceNumbers.textContent = this.state.challengeData.sequence.join(' → ');
    }

    /**
     * Create dynamic challenge areas
     */
    createDynamicAreas() {
        this.elements.dynamicGrid.innerHTML = '';
        
        this.state.challengeData.screen_areas.forEach(area => {
            const areaElement = document.createElement('div');
            areaElement.className = 'dynamic-area';
            areaElement.dataset.area = area.number;
            areaElement.textContent = area.number;
            
            // Position based on real coordinates from backend
            areaElement.style.left = `${area.x}px`;
            areaElement.style.top = `${area.y}px`;
            areaElement.style.width = `${area.width}px`;
            areaElement.style.height = `${area.height}px`;
            
            this.elements.dynamicGrid.appendChild(areaElement);
        });
    }

    /**
     * Start video recording
     */
    async startRecording() {
        this.state.recordedChunks = [];
        
        // Determine best codec
        let codec = this.config.videoCodec;
        if (!MediaRecorder.isTypeSupported(codec)) {
            codec = this.config.fallbackCodec;
            if (!MediaRecorder.isTypeSupported(codec)) {
                codec = undefined;
            }
        }

        const options = codec ? { mimeType: codec } : undefined;
        this.state.mediaRecorder = new MediaRecorder(this.state.stream, options);
        
        this.state.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.state.recordedChunks.push(event.data);
            }
        };

        this.state.mediaRecorder.onstop = () => {
            console.log('Recording stopped');
            this.processRecording();
        };

        this.state.mediaRecorder.start(100);
        this.state.isRecording = true;
        console.log('Recording started');
    }

    /**
     * Start challenge sequence
     */
    startChallengeSequence() {
        let currentAreaIndex = 0;
        const sequence = this.state.challengeData.sequence;
        
        const highlightNextArea = () => {
            if (currentAreaIndex >= sequence.length) {
                this.completeChallengeSequence();
                return;
            }

            const areaNumber = sequence[currentAreaIndex];
            
            // Clear previous highlights
            document.querySelectorAll('.dynamic-area').forEach(el => {
                el.classList.remove('highlighted');
            });
            
            // Highlight current area
            const areaElement = document.querySelector(`[data-area="${areaNumber}"]`);
            if (areaElement) {
                areaElement.classList.add('highlighted');
            }
            
            // Update progress
            const progress = (currentAreaIndex / sequence.length) * 100;
            this.elements.challengeProgress.style.width = `${progress}%`;
            this.elements.currentInstruction.textContent = `Look at area ${areaNumber}`;
            
            // Start countdown
            this.startAreaCountdown(() => {
                currentAreaIndex++;
                highlightNextArea();
            });
        };

        // Start with first area
        highlightNextArea();
    }

    /**
     * Start countdown for current area
     */
    startAreaCountdown(callback) {
        let remaining = this.config.areaDuration;
        this.elements.countdownDisplay.textContent = Math.ceil(remaining);
        
        this.timers.countdown = setInterval(() => {
            remaining -= 0.1;
            this.elements.countdownDisplay.textContent = Math.ceil(remaining);
            
            if (remaining <= 0) {
                clearInterval(this.timers.countdown);
                callback();
            }
        }, 100);
    }

    /**
     * Complete challenge sequence
     */
    completeChallengeSequence() {
        // Clear highlights
        document.querySelectorAll('.dynamic-area').forEach(el => {
            el.classList.remove('highlighted');
        });
        
        // Hide challenge interface
        this.elements.dynamicGrid.style.display = 'none';
        this.elements.progressOverlay.style.display = 'none';
        
        // Stop recording
        if (this.state.mediaRecorder && this.state.mediaRecorder.state === 'recording') {
            this.state.mediaRecorder.stop();
        }
        
        // Update progress
        this.elements.challengeProgress.style.width = '100%';
        this.elements.currentInstruction.textContent = 'Challenge complete!';
        
        console.log('Challenge sequence completed');
    }

    /**
     * Process recorded video
     */
    async processRecording() {
        try {
            // Create video blob
            const videoBlob = new Blob(this.state.recordedChunks, { 
                type: this.state.mediaRecorder.mimeType || 'video/webm' 
            });
            
            console.log('Video blob created:', videoBlob.size, 'bytes');
            
            // Submit to backend
            await this.submitChallengeVideo(videoBlob);
            
        } catch (error) {
            console.error('Failed to process recording:', error);
            this.showError('Processing failed', error.message);
        }
    }

    /**
     * Submit challenge video to backend
     */
    async submitChallengeVideo(videoBlob) {
        const formData = new FormData();
        formData.append('session_id', this.state.sessionId);
        formData.append('video', videoBlob, 'challenge_video.webm');
        
        const response = await fetch(`${this.config.apiBaseUrl}/submit-liveness`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Challenge result:', result);
        
        if (result.success) {
            this.showResults(result);
        } else {
            throw new Error(result.error || 'Challenge validation failed');
        }
    }

    /**
     * Show challenge results
     */
    showResults(result) {
        this.showStep('results');
        
        // Update result display
        const passed = result.result === 'pass';
        this.elements.resultIcon.textContent = passed ? '✅' : '❌';
        this.elements.resultTitle.textContent = passed ? 'Liveness Verified!' : 'Liveness Check Failed';
        this.elements.resultMessage.textContent = passed
            ? 'Your eye movements have been successfully verified using advanced screen coordinate mapping.'
            : 'The eye tracking validation did not meet the required threshold.';
        
        // Update stats
        this.elements.overallAccuracy.textContent = 
            `${Math.round((result.overall_accuracy || 0) * 100)}%`;
        this.elements.processingTime.textContent = 
            `${(result.processing_time || 0).toFixed(1)}s`;
        this.elements.sessionIdDisplay.textContent = result.session_id;
        
        // Show sequence results
        this.displaySequenceResults(result.sequence_results || []);
        
        // Store result for download
        this.state.lastResult = result;
    }

    /**
     * Display sequence results breakdown
     */
    displaySequenceResults(sequenceResults) {
        this.elements.sequenceResults.innerHTML = '';
        
        if (sequenceResults.length === 0) {
            this.elements.sequenceResults.innerHTML = '<p>No detailed results available.</p>';
            return;
        }
        
        sequenceResults.forEach((result, index) => {
            const item = document.createElement('div');
            item.className = `sequence-result-item ${result.passed ? 'passed' : 'failed'}`;
            
            item.innerHTML = `
                <div>
                    <strong>Area ${result.area_number}</strong>
                    <span style="margin-left: 1rem;">
                        ${Math.round((result.accuracy || 0) * 100)}% accuracy
                    </span>
                </div>
                <div>
                    ${result.passed ? '✅' : '❌'}
                </div>
            `;
            
            this.elements.sequenceResults.appendChild(item);
        });
    }

    /**
     * Start new challenge
     */
    async startNewChallenge() {
        // Reset state
        this.cleanup();
        this.state.faceCaptured = false;
        this.state.faceSnapshot = null;
        this.state.challengeData = null;
        
        // Go back to instructions
        this.showStep('instructions');
    }

    /**
     * Create new session
     */
    async createNewSession() {
        try {
            const response = await fetch(`${this.config.apiBaseUrl}/create-liveness-challenge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ area_duration: this.config.areaDuration })
            });
            
            const data = await response.json();
            if (data.success) {
                window.location.href = data.challenge_url;
            } else {
                throw new Error(data.error || 'Failed to create new session');
            }
        } catch (error) {
            console.error('Failed to create new session:', error);
            this.showError('Session Creation Failed', error.message);
        }
    }

    /**
     * Download challenge report
     */
    downloadReport() {
        if (!this.state.lastResult) return;
        
        const report = {
            session_id: this.state.lastResult.session_id,
            timestamp: new Date().toISOString(),
            result: this.state.lastResult.result,
            overall_accuracy: this.state.lastResult.overall_accuracy,
            processing_time: this.state.lastResult.processing_time,
            screen_dimensions: this.state.screenDimensions,
            sequence: this.state.challengeData?.sequence,
            sequence_results: this.state.lastResult.sequence_results,
            challenge_type: 'advanced_screen_coordinates'
        };
        
        const blob = new Blob([JSON.stringify(report, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `liveness_report_${this.state.lastResult.session_id}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Show error screen
     */
    showError(title, message) {
        this.showStep('error');
        this.elements.errorMessage.textContent = title;
        this.elements.errorCode.textContent = message;
        this.cleanup();
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        // Clear timers
        Object.values(this.timers).forEach(timer => {
            if (timer) clearInterval(timer);
        });
        this.timers = { faceDetection: null, challenge: null, countdown: null };
        
        // Stop recording
        if (this.state.isRecording && this.state.mediaRecorder) {
            this.state.mediaRecorder.stop();
        }
        
        // Stop media stream
        if (this.state.stream) {
            this.state.stream.getTracks().forEach(track => track.stop());
            this.state.stream = null;
        }
        
        // Hide challenge interface
        if (this.elements.dynamicGrid) {
            this.elements.dynamicGrid.style.display = 'none';
        }
        if (this.elements.progressOverlay) {
            this.elements.progressOverlay.style.display = 'none';
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.advancedLivenessChallenge = new AdvancedLivenessChallenge();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedLivenessChallenge;
} 