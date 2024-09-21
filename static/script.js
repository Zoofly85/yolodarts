// Function to handle calibration button clicks
async function calibrate(cameraNumber) {
    console.log(`Calibrating camera ${cameraNumber}`);
    const statusElement = document.getElementById(`calibration-status-${cameraNumber}`);
    statusElement.textContent = 'Calibrating...';
    try {
        const response = await fetch(`/calibrate/${cameraNumber}`, { method: 'POST' });
        const result = await response.json();
        statusElement.textContent = result.message;
    } catch (error) {
        console.error('Error during calibration:', error);
        statusElement.textContent = 'An error occurred during calibration. Please try again.';
    }
}

// Function to refresh camera feeds
function refreshCameraFeeds() {
    for (let i = 1; i <= 3; i++) {
        const img = document.getElementById(`camera${i}`);
        if (img) {
            const timestamp = new Date().getTime();
            const drawScoring = window.location.pathname === '/calibration' ? 'true' : 'false';
            img.src = `/video_feed/${i}?t=${timestamp}&draw_scoring=${drawScoring}`;
        }
    }
}

// Refresh camera feeds every 5 seconds
setInterval(refreshCameraFeeds, 5000);

// Initial refresh
refreshCameraFeeds();

// Function to navigate to the score page
function navigateToScore() {
    window.location.href = '/score';
}

// Function to get the current score
async function getScore() {
    try {
        const response = await fetch('/get_score');
        const data = await response.json();
        const bestScoreElement = document.getElementById('best-score');
        const dartLocationElement = document.getElementById('dart-location');
        
        // Update camera scores
        for (let i = 0; i < 3; i++) {
            const cameraScoreElement = document.getElementById(`camera-${i+1}-score`);
            cameraScoreElement.textContent = data.camera_scores[i] !== null ? data.camera_scores[i] : '-';
        }
        
        if (data.score !== null) {
            // Update best score
            bestScoreElement.textContent = data.score;
            
            // Update dart location
            dartLocationElement.textContent = `Best dart detected at: (${data.location[0].toFixed(2)}, ${data.location[1].toFixed(2)})`;
        } else {
            bestScoreElement.textContent = '-';
            dartLocationElement.textContent = 'No dart detected';
        }
    } catch (error) {
        console.error('Error getting score:', error);
    }
}

// Check for score every 2 seconds when on the score page
if (window.location.pathname === '/score') {
    setInterval(getScore, 2000);
}

// Add event listener to the play button if it exists
document.addEventListener('DOMContentLoaded', () => {
    const playButton = document.getElementById('play-button');
    if (playButton) {
        playButton.addEventListener('click', navigateToScore);
    }
});