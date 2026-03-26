// ================================
// DOM ELEMENTS
// ================================

const form = document.getElementById('predictionForm');
const userTextArea = document.getElementById('userText');
const charCountDisplay = document.getElementById('charCount');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// ================================
// CONFIGURATION
// ================================

// Use relative URL for flexibility (works with any host/port)
const API_BASE_URL = '';
const PREDICTION_ENDPOINT = '/predict';

// ================================
// EVENT LISTENERS
// ================================

document.addEventListener('DOMContentLoaded', () => {
    form.addEventListener('submit', handleFormSubmit);
    newAnalysisBtn.addEventListener('click', resetForm);
    userTextArea.addEventListener('input', updateCharCount);
});

// ================================
// CHARACTER COUNT
// ================================

function updateCharCount() {
    const count = userTextArea.value.length;
    charCountDisplay.textContent = Math.min(count, 1000);
    
    // Limit to 1000 characters
    if (count > 1000) {
        userTextArea.value = userTextArea.value.substring(0, 1000);
        charCountDisplay.textContent = 1000;
    }
}

// ================================
// FORM SUBMISSION
// ================================

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const userText = userTextArea.value.trim();
    
    if (!userText) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (userText.length < 10) {
        showError('Please enter at least 10 characters for better analysis.');
        return;
    }
    
    // Hide error and results
    hideError();
    resultsSection.classList.add('hidden');
    
    // Show loading
    loadingSpinner.classList.remove('hidden');
    form.style.opacity = '0.5';
    form.querySelector('button').disabled = true;
    
    try {
        const response = await fetch(`${API_BASE_URL}${PREDICTION_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `text=${encodeURIComponent(userText)}`
        });
        
        if (!response.ok) {
            throw new Error('Server error. Please try again.');
        }
        
        const data = await response.json();
        
        // Hide loading
        loadingSpinner.classList.add('hidden');
        form.style.opacity = '1';
        form.querySelector('button').disabled = false;
        
        // Display results
        displayResults(data);
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
        
    } catch (error) {
        console.error('Error:', error);
        loadingSpinner.classList.add('hidden');
        form.style.opacity = '1';
        form.querySelector('button').disabled = false;
        showError(error.message || 'An error occurred during analysis. Please try again.');
    }
}

// ================================
// DISPLAY RESULTS
// ================================

function displayResults(data) {
    // Parse the prediction string (e.g., "Depression Detected 😔 (Confidence: 0.85)")
    // or "No Depression 😊 (Confidence: 0.15)"
    
    // Extract severity, risk score, emotion, and suggestions from response
    const severity = extractSeverity(data.prediction);
    const riskScore = extractRiskScore(data.prediction);
    const emotion = extractEmotion(data.prediction);
    const suicideRisk = data.suicide_risk || 'N/A';
    const strategies = data.strategies || getDefaultStrategies(severity);
    
    // Update severity card
    updateSeverityCard(severity);
    
    // Update risk card
    updateRiskCard(riskScore);
    
    // Update emotion card
    updateEmotionCard(emotion);
    
    // Update suicide risk alert
    updateSuicideRiskAlert(suicideRisk);
    
    // Update strategies card
    updateStrategiesCard(strategies);
    
    // Show results section
    resultsSection.classList.remove('hidden');
}

// ================================
// EXTRACTION FUNCTIONS
// ================================

function extractSeverity(prediction) {
    if (prediction.toLowerCase().includes('no depression')) {
        return 'none';
    } else if (prediction.toLowerCase().includes('mild')) {
        return 'mild';
    } else if (prediction.toLowerCase().includes('moderate')) {
        return 'moderate';
    } else if (prediction.toLowerCase().includes('severe')) {
        return 'severe';
    }
    return 'mild';
}

function extractRiskScore(prediction) {
    const match = prediction.match(/Confidence:\s*([\d.]+)/);
    if (match) {
        const confidence = parseFloat(match[1]);
        // Convert confidence to risk score (higher confidence of depression = higher risk score)
        const riskScore = Math.round(confidence * 100);
        return Math.min(riskScore, 100);
    }
    return 50;
}

function extractEmotion(prediction) {
    if (prediction.includes('😔')) {
        return 'Sadness';
    } else if (prediction.includes('😊')) {
        return 'Contentment';
    } else if (prediction.includes('😢')) {
        return 'Sorrow';
    } else if (prediction.includes('😰')) {
        return 'Anxiety';
    }
    return 'Mixed Emotions';
}

// ================================
// UPDATE CARD FUNCTIONS
// ================================

function updateSeverityCard(severity) {
    const badge = document.getElementById('severityBadge');
    const label = document.getElementById('severityLabel');
    const description = document.getElementById('severityDescription');
    
    const severityData = {
        none: {
            label: 'No Depression Detected',
            badge: 'None',
            description: 'Based on your text, depression indicators are minimal. Keep maintaining healthy habits and continue self-care practices.',
            class: 'mild'
        },
        mild: {
            label: 'Mild Depression',
            badge: 'Mild',
            description: 'You may be experiencing some depressive symptoms. Consider reaching out to friends, family, or a mental health professional for support.',
            class: 'mild'
        },
        moderate: {
            label: 'Moderate Depression',
            badge: 'Moderate',
            description: 'Your text suggests moderate depressive symptoms. We strongly encourage speaking with a mental health professional.',
            class: 'moderate'
        },
        severe: {
            label: 'Severe Depression',
            badge: 'Severe',
            description: 'Your text indicates severe depressive symptoms. Please reach out to a mental health professional, counselor, or crisis helpline immediately.',
            class: 'severe'
        }
    };
    
    const data = severityData[severity] || severityData.mild;
    
    badge.textContent = data.badge;
    badge.className = `severity-badge ${data.class}`;
    label.textContent = data.label;
    description.textContent = data.description;
}

function updateRiskCard(riskScore) {
    const gaugeFill = document.getElementById('gaugeFill');
    const riskScoreDisplay = document.getElementById('riskScore');
    const interpretation = document.getElementById('riskInterpretation');
    
    // Animate gauge
    setTimeout(() => {
        gaugeFill.style.width = riskScore + '%';
    }, 100);
    
    riskScoreDisplay.textContent = riskScore;
    
    let interpretationText = '';
    if (riskScore < 30) {
        interpretationText = '✅ Your mental health risk appears to be low. Continue with your wellness routine and stay connected with loved ones.';
    } else if (riskScore < 60) {
        interpretationText = '⚠️ Moderate risk detected. Consider speaking with a counselor or therapist to discuss your feelings and develop coping strategies.';
    } else {
        interpretationText = '🚨 High risk detected. Please prioritize reaching out to a mental health professional or crisis support right away.';
    }
    
    interpretation.textContent = interpretationText;
}

function updateEmotionCard(emotion) {
    const emoji = document.getElementById('emotionEmoji');
    const emotionText = document.getElementById('emotionText');
    const description = document.getElementById('emotionDescription');
    
    const emotionData = {
        'Sadness': {
            emoji: '😢',
            description: 'You\'re experiencing sadness. It\'s okay to feel this way. Consider connecting with supportive people or activities you enjoy.'
        },
        'Contentment': {
            emoji: '😊',
            description: 'You seem to be in a positive emotional state. Great! Keep nurturing what brings you joy and peace.'
        },
        'Sorrow': {
            emoji: '😔',
            description: 'Deep sorrow and grief can be overwhelming. Please reach out to someone you trust or a professional counselor.'
        },
        'Anxiety': {
            emoji: '😰',
            description: 'Anxiety is present in your thoughts. Try grounding techniques, deep breathing, or reach out to a mental health professional.'
        },
        'Mixed Emotions': {
            emoji: '🎭',
            description: 'You\'re experiencing a complex mix of emotions. This is normal during challenging times. Consider journaling or speaking with someone.'
        }
    };
    
    const data = emotionData[emotion] || emotionData['Mixed Emotions'];
    
    emoji.textContent = data.emoji;
    emotionText.textContent = emotion;
    description.textContent = data.description;
}

function updateSuicideRiskAlert(suicideRisk) {
    const alertCard = document.getElementById('suicideRiskAlert');
    
    if (suicideRisk && (suicideRisk.toLowerCase().includes('high') || suicideRisk.toLowerCase().includes('severe'))) {
        alertCard.classList.remove('hidden');
    } else {
        alertCard.classList.add('hidden');
    }
}

function updateStrategiesCard(strategies) {
    const list = document.getElementById('strategiesList');
    list.innerHTML = '';
    
    const strategiesToDisplay = Array.isArray(strategies) ? strategies : [];
    
    if (strategiesToDisplay.length === 0) {
        strategiesToDisplay.push(
            'Take a few deep breaths and practice grounding techniques',
            'Reach out to a trusted friend or family member',
            'Engage in physical activity or gentle exercise',
            'Practice self-compassion and be kind to yourself',
            'Consult with a mental health professional'
        );
    }
    
    strategiesToDisplay.forEach(strategy => {
        const li = document.createElement('li');
        li.textContent = strategy;
        list.appendChild(li);
    });
}

// ================================
// DEFAULT STRATEGIES BY SEVERITY
// ================================

function getDefaultStrategies(severity) {
    const strategies = {
        none: [
            'Continue maintaining regular sleep schedule',
            'Stay physically active with exercise you enjoy',
            'Nurture your relationships and social connections',
            'Practice mindfulness or meditation daily',
            'Engage in hobbies and activities you love'
        ],
        mild: [
            'Talk to someone you trust about your feelings',
            'Practice regular exercise (at least 30 minutes daily)',
            'Maintain a healthy sleep routine',
            'Try journaling to express your emotions',
            'Consider speaking with a therapist or counselor'
        ],
        moderate: [
            'Seek professional mental health support urgently',
            'Reach out to trusted friends or family members',
            'Practice self-care activities daily',
            'Avoid isolating yourself; stay connected',
            'Consider professional medication consultation if needed'
        ],
        severe: [
            'Contact a mental health crisis hotline immediately',
            'Reach out to a trusted mental health professional',
            'Call 988 (Suicide & Crisis Lifeline) if in the US',
            'Tell someone close to you how you\'re feeling',
            'Avoid alcohol and limit caffeine intake'
        ]
    };
    
    return strategies[severity] || strategies.mild;
}

// ================================
// ERROR HANDLING
// ================================

function showError(message) {
    errorMessage.textContent = '❌ ' + message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// ================================
// RESET FORM
// ================================

function resetForm() {
    form.reset();
    userTextArea.value = '';
    charCountDisplay.textContent = '0';
    resultsSection.classList.add('hidden');
    hideError();
    loadingSpinner.classList.add('hidden');
    
    // Scroll back to form
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    userTextArea.focus();
}

// ================================
// UTILITY FUNCTIONS
// ================================

// Smooth scroll helper
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
