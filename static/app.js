/**
 * Eye Tracking Liveness Challenge Application
 * Handles camera access, video recording, API communication, and UI management
 */

class LivenessChallenge {
    constructor() {
        // Configuration
        this.config = {
            apiBaseUrl: 'http://localhost:8000',
            videoDuration: null, // Will be set based on sequence
            areaDuration: 3.0, // seconds per area
            videoCodec: 'video/webm;codecs=vp9',
            fallbackCodec: 'video/webm',
            maxRetries: 3,
            debugMode: false
        };

        // State management
        this.state = {
            currentScreen: 'start',
            sessionId: null,
            challengeUrl: null,
            sequence: [],
            currentAreaIndex: 0,
            isRecording: false,
            mediaRecorder: null,
            recordedChunks: [],
            stream: null,
            challengeStartTime: null,
            retryCount: 0
        };

        // DOM elements
        this.elements = {};
        
        // Face detection state
        this.faceDetection = {
            isDetected: false,
            confidence: 0,
            position: null,
            lastDetectionTime: 0
        };
        
        // Timers
        this.timers = {
            countdown: null,
            challenge: null,
            areaTimer: null,
            faceDetection: null
        };

        this.initializeApp();
    }

    /**
     * Initialize the application
     */
    async initializeApp() {
        try {
            this.bindElements();
            this.bindEvents();
            this.updateDebugPanel();
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.showError('Initialization failed', error.message);
        }
    }

    /**
     * Bind DOM elements
     */
    bindElements() {
        this.elements = {
            // Screens
            startScreen: document.getElementById('start-screen'),
            challengeScreen: document.getElementById('challenge-screen'),
            processingScreen: document.getElementById('processing-screen'),
            resultsScreen: document.getElementById('results-screen'),
            errorScreen: document.getElementById('error-screen'),

            // Start screen
            startChallenge: document.getElementById('start-challenge'),
            cameraStatus: document.getElementById('camera-status'),
            cameraIndicator: document.getElementById('camera-indicator'),

            // Challenge screen
            video: document.getElementById('video'),
            challengeGrid: document.getElementById('challenge-grid'),
            progressBar: document.querySelector('#progress-bar .progress-fill'),
            progressText: document.getElementById('progress-text'),
            currentInstruction: document.getElementById('current-instruction'),
            sequenceNumbers: document.getElementById('sequence-numbers'),
            countdown: document.getElementById('countdown'),
            faceGuideOverlay: document.querySelector('.face-guide-overlay'),
            faceStatus: document.querySelector('.face-status'),
            faceStatusIcon: document.getElementById('face-status-icon'),
            faceStatusText: document.getElementById('face-status-text'),

            // Processing screen
            processingStatus: document.getElementById('processing-status'),
            uploadProgress: document.querySelector('#upload-progress .progress-fill'),

            // Results screen
            resultIcon: document.getElementById('result-icon'),
            resultTitle: document.getElementById('result-title'),
            resultMessage: document.getElementById('result-message'),
            overallAccuracy: document.getElementById('overall-accuracy'),
            processingTime: document.getElementById('processing-time'),
            sessionIdDisplay: document.getElementById('session-id'),
            sequenceResults: document.getElementById('sequence-results'),
            tryAgain: document.getElementById('try-again'),
            downloadReport: document.getElementById('download-report'),

            // Error screen
            errorMessage: document.getElementById('error-message'),
            errorCode: document.getElementById('error-code'),
            retryChallenge: document.getElementById('retry-challenge'),
            backToStart: document.getElementById('back-to-start'),

            // Debug
            debugPanel: document.getElementById('debug-panel'),
            debugContent: document.getElementById('debug-content')
        };
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Start screen
        this.elements.createChallenge = document.getElementById('create-challenge');
        this.elements.challengeUrlDisplay = document.getElementById('challenge-url-display');
        this.elements.challengeUrlText = document.getElementById('challenge-url-text');
        this.elements.openChallenge = document.getElementById('open-challenge');
        this.elements.copyUrl = document.getElementById('copy-url');
        
        this.elements.createChallenge.addEventListener('click', () => this.createNewChallenge());
        this.elements.openChallenge.addEventListener('click', () => this.openChallengeUrl());
        this.elements.copyUrl.addEventListener('click', () => this.copyChallengeUrl());

        // Results screen
        this.elements.tryAgain.addEventListener('click', () => this.resetToStart());
        this.elements.downloadReport.addEventListener('click', () => this.downloadReport());

        // Error screen
        this.elements.retryChallenge.addEventListener('click', () => this.startChallenge());
        this.elements.backToStart.addEventListener('click', () => this.resetToStart());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.resetToStart();
            } else if (e.key === 'Enter' && this.state.currentScreen === 'start') {
                this.startChallenge();
            } else if (e.key === 'F12') {
                this.toggleDebugPanel();
            }
        });

        // Window events
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });

        // Visibility change (tab switching)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.state.isRecording) {
                console.warn('Tab became hidden during recording');
            }
        });
    }

    /**
     * Check camera access and permissions
     */
    async checkCameraAccess() {
        try {
            this.updateCameraStatus('Requesting camera access...', '📷');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                }, 
                audio: false 
            });
            
            // Test stream
            this.elements.video.srcObject = stream;
            
            // Stop the test stream
            stream.getTracks().forEach(track => track.stop());
            this.elements.video.srcObject = null;
            
            this.updateCameraStatus('Camera ready', '✅');
            this.elements.startChallenge.disabled = false;
            
        } catch (error) {
            console.error('Camera access failed:', error);
            let message = 'Camera access denied';
            let icon = '❌';
            
            if (error.name === 'NotAllowedError') {
                message = 'Camera permission denied. Please allow camera access and refresh.';
            } else if (error.name === 'NotFoundError') {
                message = 'No camera found. Please ensure a camera is connected.';
            } else if (error.name === 'NotReadableError') {
                message = 'Camera is in use by another application.';
            }
            
            this.updateCameraStatus(message, icon);
        }
    }

    /**
     * Update camera status display
     */
    updateCameraStatus(message, icon) {
        this.elements.cameraStatus.textContent = message;
        this.elements.cameraIndicator.textContent = icon;
    }

    /**
     * Start the liveness challenge
     */
    async startChallenge() {
        try {
            this.state.retryCount++;
            this.showScreen('processing');
            this.elements.processingStatus.textContent = 'Initializing challenge...';
            
            // Initialize challenge session
            const sessionData = await this.initiateLivenessChallenge();
            this.state.sessionId = sessionData.session_id;
            this.state.sequence = sessionData.sequence;
            this.config.videoDuration = sessionData.total_duration;
            
            // Setup camera and recording
            await this.setupCamera();
            await this.setupRecording();
            
            // Start the challenge
            this.showScreen('challenge');
            this.startChallengeSequence();
            
        } catch (error) {
            console.error('Failed to start challenge:', error);
            this.showError('Failed to start challenge', error.message);
        }
    }

    /**
     * Initiate liveness challenge with backend
     */
    async initiateLivenessChallenge() {
        const response = await fetch(`${this.config.apiBaseUrl}/initiate-liveness`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                area_duration: this.config.areaDuration
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to initiate challenge');
        }

        this.log('Challenge initiated:', data);
        return data;
    }

    /**
     * Create new liveness challenge
     */
    async createNewChallenge() {
        try {
            this.elements.createChallenge.disabled = true;
            this.elements.createChallenge.textContent = 'Creating Simple Challenge...';
            
            const response = await fetch(`${this.config.apiBaseUrl}/create-simple-challenge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Failed to create challenge');
            }

            // Store challenge URL
            this.state.challengeUrl = data.challenge_url;
            this.state.sessionId = data.session_id;
            
            // Display URL
            this.elements.challengeUrlText.textContent = data.challenge_url;
            this.elements.challengeUrlDisplay.style.display = 'block';
            
            this.log('Challenge URL created:', data);
            
        } catch (error) {
            console.error('Failed to create challenge:', error);
            this.showError('Challenge Creation Failed', error.message);
        } finally {
            this.elements.createChallenge.disabled = false;
            this.elements.createChallenge.textContent = 'Start Random Challenge';
        }
    }

    /**
     * Open challenge URL in new tab
     */
    openChallengeUrl() {
        if (this.state.challengeUrl) {
            window.open(this.state.challengeUrl, '_blank');
        }
    }

    /**
     * Copy challenge URL to clipboard
     */
    async copyChallengeUrl() {
        if (this.state.challengeUrl) {
            try {
                await navigator.clipboard.writeText(this.state.challengeUrl);
                
                // Temporary feedback
                const originalText = this.elements.copyUrl.textContent;
                this.elements.copyUrl.textContent = 'Copied!';
                this.elements.copyUrl.style.background = 'var(--success-color)';
                
                setTimeout(() => {
                    this.elements.copyUrl.textContent = originalText;
                    this.elements.copyUrl.style.background = '';
                }, 2000);
                
            } catch (error) {
                console.error('Failed to copy URL:', error);
                // Fallback for older browsers
                this.fallbackCopyToClipboard(this.state.challengeUrl);
            }
        }
    }

    /**
     * Fallback copy method for older browsers
     */
    fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            const originalText = this.elements.copyUrl.textContent;
            this.elements.copyUrl.textContent = 'Copied!';
            setTimeout(() => {
                this.elements.copyUrl.textContent = originalText;
            }, 2000);
        } catch (error) {
            console.error('Fallback copy failed:', error);
        }
        
        document.body.removeChild(textArea);
    }

    /**
     * Setup camera stream
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
        
        this.elements.video.srcObject = this.state.stream;
        
        // Start face detection monitoring
        this.startFaceDetectionMonitoring();
    }

    /**
     * Start face detection monitoring
     */
    startFaceDetectionMonitoring() {
        // Simple face detection based on video stream activity
        const checkFaceDetection = () => {
            if (!this.elements.video || !this.state.stream) {
                this.updateFaceStatus(false, 'No video stream');
                return;
            }
            
            const video = this.elements.video;
            
            // Check if video is playing and has content
            if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
                // Simulate face detection - in a real implementation you'd use
                // MediaPipe, face-api.js, or similar for actual face detection
                const now = Date.now();
                const hasVideo = !video.paused && !video.ended && video.currentTime > 0;
                
                if (hasVideo) {
                    this.faceDetection.isDetected = true;
                    this.faceDetection.lastDetectionTime = now;
                    this.updateFaceStatus(true, 'Face detected - Good positioning!');
                } else {
                    this.updateFaceStatus(false, 'Please ensure your face is visible');
                }
            } else {
                this.updateFaceStatus(false, 'Loading camera...');
            }
        };
        
        // Start monitoring
        this.timers.faceDetection = setInterval(checkFaceDetection, 500);
    }
    
    /**
     * Update face detection status UI
     */
    updateFaceStatus(detected, message) {
        this.faceDetection.isDetected = detected;
        
        if (!this.elements.faceStatus) return;
        
        // Update status text and icon
        this.elements.faceStatusText.textContent = message;
        
        // Update visual state
        this.elements.faceStatus.classList.remove('detected', 'warning');
        
        if (detected) {
            this.elements.faceStatus.classList.add('detected');
            this.elements.faceStatusIcon.textContent = '✅';
            // Hide face guide overlay when face is detected
            if (this.elements.faceGuideOverlay) {
                this.elements.faceGuideOverlay.style.opacity = '0.3';
            }
        } else {
            this.elements.faceStatus.classList.add('warning');
            this.elements.faceStatusIcon.textContent = '⚠️';
            // Show face guide overlay when no face detected
            if (this.elements.faceGuideOverlay) {
                this.elements.faceGuideOverlay.style.opacity = '1';
            }
                 }
     }
     
    /**
     * Check face positioning before starting challenge
     */
    checkFacePositioningBeforeStart() {
        let countdown = 3;
        this.elements.currentInstruction.textContent = `Challenge starts in ${countdown} seconds...`;
        
        const positioningCheck = setInterval(() => {
            countdown--;
            
            if (countdown > 0) {
                this.elements.currentInstruction.textContent = `Challenge starts in ${countdown} seconds...`;
                
                // Provide positioning feedback
                if (!this.faceDetection.isDetected) {
                    this.elements.currentInstruction.textContent = 
                        `Position your face properly! Starting in ${countdown}...`;
                }
            } else {
                clearInterval(positioningCheck);
                this.elements.currentInstruction.textContent = 'Get ready...';
            }
        }, 1000);
    }

    /**
     * Setup media recording
     */
    async setupRecording() {
        this.state.recordedChunks = [];
        
        // Determine best codec
        let codec = this.config.videoCodec;
        if (!MediaRecorder.isTypeSupported(codec)) {
            codec = this.config.fallbackCodec;
            if (!MediaRecorder.isTypeSupported(codec)) {
                codec = undefined; // Use default
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
            this.log('Recording stopped');
        };

        this.log('Recording setup complete:', { codec, options });
    }

    /**
     * Start the challenge sequence
     */
    startChallengeSequence() {
        this.state.currentAreaIndex = 0;
        this.state.challengeStartTime = Date.now();
        
        // Update UI
        this.elements.sequenceNumbers.textContent = this.state.sequence.join(' → ');
        this.elements.currentInstruction.textContent = 'Get ready...';
        
        // Check face positioning before starting
        this.checkFacePositioningBeforeStart();
        
        // Start recording
        this.startRecording();
        
        // Start sequence with delay
        setTimeout(() => {
            this.highlightNextArea();
        }, 1000);
    }

    /**
     * Start video recording
     */
    startRecording() {
        if (this.state.mediaRecorder && this.state.mediaRecorder.state === 'inactive') {
            this.state.mediaRecorder.start(100); // Record in 100ms chunks
            this.state.isRecording = true;
            this.log('Recording started');
        }
    }

    /**
     * Stop video recording
     */
    stopRecording() {
        if (this.state.mediaRecorder && this.state.mediaRecorder.state === 'recording') {
            this.state.mediaRecorder.stop();
            this.state.isRecording = false;
            this.log('Recording stop requested');
        }
    }

    /**
     * Highlight the next area in sequence
     */
    highlightNextArea() {
        if (this.state.currentAreaIndex >= this.state.sequence.length) {
            this.completeChallengeSequence();
            return;
        }

        const areaNumber = this.state.sequence[this.state.currentAreaIndex];
        const areaElement = document.querySelector(`[data-area="${areaNumber}"]`);
        
        // Clear previous highlights
        document.querySelectorAll('.grid-area').forEach(el => {
            el.classList.remove('highlighted');
        });
        
        // Highlight current area
        areaElement.classList.add('highlighted');
        
        // Update UI
        this.elements.currentInstruction.textContent = `Look at area ${areaNumber}`;
        this.updateProgress();
        
        // Start countdown
        this.startAreaCountdown(() => {
            this.state.currentAreaIndex++;
            this.highlightNextArea();
        });
    }

    /**
     * Start countdown for current area
     */
    startAreaCountdown(callback) {
        let remaining = this.config.areaDuration;
        this.elements.countdown.textContent = Math.ceil(remaining);
        
        this.timers.countdown = setInterval(() => {
            remaining -= 0.1;
            this.elements.countdown.textContent = Math.ceil(remaining);
            
            if (remaining <= 0) {
                clearInterval(this.timers.countdown);
                callback();
            }
        }, 100);
    }

    /**
     * Update progress bar
     */
    updateProgress() {
        const progress = (this.state.currentAreaIndex / this.state.sequence.length) * 100;
        this.elements.progressBar.style.width = `${progress}%`;
        this.elements.progressText.textContent = 
            `Area ${this.state.currentAreaIndex + 1} of ${this.state.sequence.length}`;
    }

    /**
     * Complete the challenge sequence
     */
    completeChallengeSequence() {
        // Clear highlights
        document.querySelectorAll('.grid-area').forEach(el => {
            el.classList.remove('highlighted');
        });
        
        // Stop recording
        this.stopRecording();
        
        // Update UI
        this.elements.currentInstruction.textContent = 'Challenge complete!';
        this.elements.progressBar.style.width = '100%';
        this.elements.progressText.textContent = 'Processing...';
        
        // Wait for recording to stop and process
        setTimeout(() => {
            this.processRecording();
        }, 1000);
    }

    /**
     * Process the recorded video
     */
    async processRecording() {
        try {
            this.showScreen('processing');
            this.elements.processingStatus.textContent = 'Preparing video...';
            
            // Create video blob
            const videoBlob = new Blob(this.state.recordedChunks, { 
                type: this.state.mediaRecorder.mimeType || 'video/webm' 
            });
            
            this.log('Video blob created:', { 
                size: videoBlob.size, 
                type: videoBlob.type 
            });
            
            // Submit to backend
            await this.submitLivenessChallenge(videoBlob);
            
        } catch (error) {
            console.error('Failed to process recording:', error);
            this.showError('Processing failed', error.message);
        }
    }

    /**
     * Submit liveness challenge to backend
     */
    async submitLivenessChallenge(videoBlob) {
        const formData = new FormData();
        formData.append('session_id', this.state.sessionId);
        formData.append('video', videoBlob, 'challenge_video.webm');
        
        this.elements.processingStatus.textContent = 'Uploading video...';
        this.elements.uploadProgress.style.width = '0%';
        
        try {
            const response = await this.uploadWithProgress(`${this.config.apiBaseUrl}/submit-liveness`, formData);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            this.log('Challenge result:', result);
            
            if (result.success) {
                this.showResults(result);
            } else {
                throw new Error(result.error || 'Challenge validation failed');
            }
            
        } catch (error) {
            console.error('Submission failed:', error);
            throw error;
        }
    }

    /**
     * Upload with progress tracking
     */
    async uploadWithProgress(url, formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    this.elements.uploadProgress.style.width = `${percentComplete}%`;
                    this.elements.processingStatus.textContent = 
                        `Uploading... ${Math.round(percentComplete)}%`;
                }
            });
            
            xhr.addEventListener('load', () => {
                this.elements.processingStatus.textContent = 'Analyzing video...';
                this.elements.uploadProgress.style.width = '100%';
                
                // Create a Response-like object with proper json() method
                const response = {
                    ok: xhr.status >= 200 && xhr.status < 300,
                    status: xhr.status,
                    statusText: xhr.statusText,
                    json: async () => {
                        try {
                            return JSON.parse(xhr.responseText);
                        } catch (e) {
                            throw new Error('Invalid JSON response');
                        }
                    }
                };
                
                resolve(response);
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Upload failed'));
            });
            
            xhr.open('POST', url);
            xhr.send(formData);
        });
    }

    /**
     * Show challenge results
     */
    showResults(result) {
        this.showScreen('results');
        
        // Update result display
        const passed = result.result === 'pass';
        this.elements.resultIcon.textContent = passed ? '✅' : '❌';
        this.elements.resultTitle.textContent = passed ? 'Liveness Verified!' : 'Liveness Check Failed';
        this.elements.resultMessage.textContent = passed
            ? 'Your eye movements have been successfully verified.'
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
            sequence: this.state.sequence,
            sequence_results: this.state.lastResult.sequence_results,
            challenge_duration: this.config.videoDuration,
            area_duration: this.config.areaDuration
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
        this.showScreen('error');
        this.elements.errorMessage.textContent = title;
        this.elements.errorCode.textContent = message;
        this.cleanup();
    }

    /**
     * Show specific screen
     */
    showScreen(screenName) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show target screen
        const targetScreen = document.getElementById(`${screenName}-screen`);
        if (targetScreen) {
            targetScreen.classList.add('active');
            this.state.currentScreen = screenName;
        }
        
        // Show face guide overlay when entering challenge screen
        if (screenName === 'challenge' && this.elements.faceGuideOverlay) {
            this.elements.faceGuideOverlay.style.opacity = '1';
            this.updateFaceStatus(false, 'Position your face in the outline');
        }
    }

    /**
     * Reset to start screen
     */
    resetToStart() {
        this.cleanup();
        this.state = {
            ...this.state,
            sessionId: null,
            sequence: [],
            currentAreaIndex: 0,
            isRecording: false,
            mediaRecorder: null,
            recordedChunks: [],
            challengeStartTime: null
        };
        this.showScreen('start');
        this.checkCameraAccess();
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        // Clear timers
        Object.values(this.timers).forEach(timer => {
            if (timer) clearInterval(timer);
        });
        this.timers = { countdown: null, challenge: null, areaTimer: null, faceDetection: null };
        
        // Stop recording
        if (this.state.isRecording) {
            this.stopRecording();
        }
        
        // Stop media stream
        if (this.state.stream) {
            this.state.stream.getTracks().forEach(track => track.stop());
            this.state.stream = null;
        }
        
        // Clear video
        if (this.elements.video) {
            this.elements.video.srcObject = null;
        }
        
        // Clear highlights
        document.querySelectorAll('.grid-area').forEach(el => {
            el.classList.remove('highlighted');
        });
    }

    /**
     * Toggle debug panel
     */
    toggleDebugPanel() {
        this.config.debugMode = !this.config.debugMode;
        this.elements.debugPanel.style.display = this.config.debugMode ? 'block' : 'none';
        this.updateDebugPanel();
    }

    /**
     * Update debug panel
     */
    updateDebugPanel() {
        if (!this.config.debugMode) return;
        
        const debugInfo = {
            state: this.state,
            config: this.config,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            mediaDevices: {
                available: !!navigator.mediaDevices,
                getUserMedia: !!navigator.mediaDevices?.getUserMedia
            }
        };
        
        this.elements.debugContent.textContent = JSON.stringify(debugInfo, null, 2);
    }

    /**
     * Log messages (with debug mode)
     */
    log(...args) {
        if (this.config.debugMode) {
            console.log('[LivenessChallenge]', ...args);
        }
        this.updateDebugPanel();
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.livenessChallenge = new LivenessChallenge();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LivenessChallenge;
} 