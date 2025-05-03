document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const promptInput = document.getElementById('prompt');
    const maxTokensInput = document.getElementById('maxTokens');
    const temperatureInput = document.getElementById('temperature');
    const generateBtn = document.getElementById('generateBtn');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const output = document.getElementById('output');
    const resultCard = document.getElementById('resultCard');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Initialize with focus on prompt
    promptInput.focus();

    // Generate text
    generateBtn.addEventListener('click', async function() {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        // Show loading spinner briefly
        loadingSpinner.style.display = 'block';
        resultCard.style.display = 'none';
        
        // Prepare output area
        setTimeout(() => {
            loadingSpinner.style.display = 'none';
            resultCard.style.display = 'block';
            output.textContent = '';
            output.classList.add('typing-animation');
        }, 500);

        try {
            // Send request to streaming endpoint
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: prompt,
                    max_tokens: parseInt(maxTokensInput.value),
                    temperature: parseFloat(temperatureInput.value)
                })
            });

            // Process the streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            // Read the stream
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                // Decode and process the chunk
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n');
                
                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        try {
                            const data = JSON.parse(line.substring(5).trim());
                            
                            if (data.token) {
                                // Append token to output
                                output.textContent += data.token;
                            }
                            
                            if (data.done) {
                                // Generation complete
                                output.classList.remove('typing-animation');
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error:', error);
            loadingSpinner.style.display = 'none';
            output.textContent = 'An error occurred while generating text. Please try again.';
            resultCard.style.display = 'block';
        }
    });

    // Clear prompt
    clearBtn.addEventListener('click', function() {
        promptInput.value = '';
        promptInput.focus();
    });

    // Copy output to clipboard
    copyBtn.addEventListener('click', function() {
        const textToCopy = output.textContent;
        
        if (!textToCopy) return;
        
        navigator.clipboard.writeText(textToCopy).then(function() {
            // Change button text temporarily
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            
            setTimeout(function() {
                copyBtn.innerHTML = originalText;
            }, 2000);
        }).catch(function(err) {
            console.error('Could not copy text: ', err);
            alert('Failed to copy to clipboard');
        });
    });

    // Allow generating by pressing Ctrl+Enter in the textarea
    promptInput.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            generateBtn.click();
        }
    });
}); 