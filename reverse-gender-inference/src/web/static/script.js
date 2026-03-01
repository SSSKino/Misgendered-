// JavaScript for Reverse Gender Inference Detection System

class ReverseInferenceApp {
    constructor() {
        this.selectedModels = new Set();
        this.availableModels = [];
        this.evaluationId = null;
        this.statusCheckInterval = null;
        this.currentResults = null;
        
        this.initializeEventListeners();
        this.loadModels();
    }

    initializeEventListeners() {
        // Start evaluation button
        document.getElementById('start-evaluation').addEventListener('click', () => {
            this.startEvaluation();
        });

        // Stop evaluation button
        document.getElementById('stop-evaluation').addEventListener('click', () => {
            this.stopEvaluation();
        });

        // Tab navigation
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Export buttons
        document.getElementById('export-results')?.addEventListener('click', () => {
            this.exportResults();
        });

        document.getElementById('save-config')?.addEventListener('click', () => {
            this.saveConfiguration();
        });

        // API key input listeners
        const apiKeyFields = ['openai_api_key', 'anthropic_api_key', 'dashscope_api_key', 'deepseek_api_key'];
        apiKeyFields.forEach(fieldId => {
            document.getElementById(fieldId)?.addEventListener('input', () => {
                this.renderModelSelection(); // Re-render model selection when API keys change
            });
        });
    }

    async loadModels() {
        try {
            console.log('Loading models from API...');
            const response = await fetch('/api/models');
            this.availableModels = await response.json();
            console.log('Loaded models:', this.availableModels);
            
            this.renderModelSelection();
        } catch (error) {
            this.showStatus('加载模型列表失败: ' + error.message, 'error');
            console.error('Failed to load models:', error);
        }
    }

    renderModelSelection() {
        const container = document.getElementById('model-selection');
        container.innerHTML = '';
        
        // Check available models from backend
        if (!this.availableModels || this.availableModels.length === 0) {
            // Show message when no models are available
            const noModelsDiv = document.createElement('div');
            noModelsDiv.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #e74c3c; border: 2px dashed #e74c3c; border-radius: 8px; margin: 15px 0;">
                    <h4 style="margin: 0 0 10px 0;">⚠️ 没有可用的模型</h4>
                    <p style="margin: 0; font-size: 14px;">请先输入有效的API密钥，然后点击下方按钮注册模型。<br>
                    支持的API：OpenAI, Anthropic, DashScope(Qwen), DeepSeek</p>
                </div>
            `;
            container.appendChild(noModelsDiv);
        }
        
        // Define API key mappings for UI
        const keyFieldMap = {
            'gpt-4o': 'openai_api_key',
            'claude-4-sonnet': 'anthropic_api_key',
            'qwen-turbo': 'dashscope_api_key',
            'qwen-2.5-72b': 'dashscope_api_key',
            'deepseek-v3': 'deepseek_api_key'
        };
        
        let hasAnyModel = false;
        
        // Render actual available models from backend
        if (this.availableModels && this.availableModels.length > 0) {
            this.availableModels.forEach(model => {
                hasAnyModel = true;
                const modelItem = document.createElement('div');
                modelItem.className = 'checkbox-item';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `model-${model.name}`;
                checkbox.value = model.name;
                
                const label = document.createElement('label');
                label.htmlFor = checkbox.id;
                label.style.cursor = 'pointer';
                label.style.margin = '0';
                label.style.display = 'flex';
                label.style.alignItems = 'center';
                label.style.width = '100%';
                
                // Determine model status
                const isDemo = model.provider === 'Demo';
                const statusText = isDemo ? '(演示模型)' : '(真实API)';
                const statusColor = isDemo ? '#e67e22' : '#2ecc71';
                
                label.innerHTML = `
                    <span style="font-weight: 600; margin-right: 8px;">${model.name}</span>
                    <span style="color: #666; font-size: 12px;">${model.description}</span>
                    <span style="color: ${statusColor}; font-size: 11px; margin-left: auto;">${statusText}</span>
                `;
                
                checkbox.addEventListener('change', () => {
                    this.toggleModelSelection(model.name, checkbox.checked);
                });
                
                modelItem.appendChild(checkbox);
                modelItem.appendChild(label);
                container.appendChild(modelItem);
            });
        }

        // Add registration and clear buttons for API keys
        const registerDiv = document.createElement('div');
        registerDiv.style.margin = '15px 0';
        registerDiv.innerHTML = `
            <div style="margin-bottom: 10px;">
                <button id="register-models-btn" style="
                    padding: 8px 16px; 
                    background: #3498db; 
                    color: white; 
                    border: none; 
                    border-radius: 4px; 
                    cursor: pointer;
                    font-size: 14px;
                    margin-right: 10px;
                ">注册API密钥并刷新模型列表</button>
                <button id="clear-models-btn" style="
                    padding: 8px 16px; 
                    background: #e74c3c; 
                    color: white; 
                    border: none; 
                    border-radius: 4px; 
                    cursor: pointer;
                    font-size: 14px;
                ">清理所有模型</button>
            </div>
            <p style="margin: 0; color: #666; font-size: 12px;">
                输入API密钥后点击注册按钮。如遇问题可先清理所有模型。
            </p>
        `;
        container.appendChild(registerDiv);
        
        // Add click handlers
        document.getElementById('register-models-btn').addEventListener('click', () => {
            this.registerModelsWithApiKeys();
        });
        
        document.getElementById('clear-models-btn').addEventListener('click', () => {
            this.clearAllModels();
        });
        
        // Update start button state
        const startButton = document.getElementById('start-evaluation');
        startButton.disabled = !hasAnyModel || this.selectedModels.size === 0;
    }

    checkApiKeyAvailable(keyField) {
        const keyInput = document.getElementById(keyField);
        return keyInput && keyInput.value.trim().length > 0;
    }

    toggleModelSelection(modelName, selected) {
        if (selected) {
            this.selectedModels.add(modelName);
        } else {
            this.selectedModels.delete(modelName);
        }
        
        // Update start button state
        const startButton = document.getElementById('start-evaluation');
        startButton.disabled = this.selectedModels.size === 0;
        
        // Update parameter suggestions based on selected models
        if (this.selectedModels.size > 0) {
            this.updateParameterSuggestions();
        }
    }

    getApiKeys() {
        const apiKeys = {};
        const keyMappings = {
            'openai_api_key': 'OPENAI_API_KEY',
            'anthropic_api_key': 'ANTHROPIC_API_KEY',
            'dashscope_api_key': 'DASHSCOPE_API_KEY',
            'deepseek_api_key': 'DEEPSEEK_API_KEY'
        };

        for (const [inputId, envName] of Object.entries(keyMappings)) {
            const input = document.getElementById(inputId);
            if (input && input.value.trim()) {
                apiKeys[envName] = input.value.trim();
            }
        }
        
        return apiKeys;
    }

    getEvaluationConfig() {
        return {
            selected_models: Array.from(this.selectedModels),
            api_keys: this.getApiKeys(),
            test_scale: parseInt(document.getElementById('test-scale').value),
            random_seed: parseInt(document.getElementById('random-seed').value),
            batch_size: parseInt(document.getElementById('batch-size').value),
            max_concurrent: parseInt(document.getElementById('max-concurrent').value),
            timeout: parseFloat(document.getElementById('timeout').value)
        };
    }

    async clearAllModels() {
        try {
            this.showStatus('正在清理所有模型...', 'info');
            
            const response = await fetch('/api/models/clear', {
                method: 'POST'
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            this.showStatus(`已清理 ${result.cleared_count} 个模型`, 'success');
            
            // Clear frontend selection state
            this.selectedModels.clear();
            
            // Refresh model list
            await this.loadModels();
            
        } catch (error) {
            this.showStatus('清理模型失败: ' + error.message, 'error');
            console.error('Failed to clear models:', error);
        }
    }

    async registerModelsWithApiKeys() {
        const apiKeys = this.getApiKeys();
        console.log('API Keys to register:', Object.keys(apiKeys));
        
        if (Object.keys(apiKeys).length === 0) {
            this.showStatus('请先输入至少一个API密钥', 'warning');
            return;
        }

        try {
            // First clear existing models to avoid conflicts
            this.showStatus('正在清理现有模型...', 'info');
            await fetch('/api/models/clear', { method: 'POST' });
            
            // Clear frontend selection state
            this.selectedModels.clear();
            
            this.showStatus('正在注册模型...', 'info');
            console.log('Sending registration request...');
            
            const response = await fetch('/api/models/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(apiKeys)
            });
            
            console.log('Registration response status:', response.status);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Registration result:', result);
            this.showStatus(`成功注册 ${result.new_models_count} 个模型`, 'success');
            
            // Refresh model list
            console.log('Refreshing model list...');
            await this.loadModels();
            
        } catch (error) {
            this.showStatus('注册模型失败: ' + error.message, 'error');
            console.error('Failed to register models:', error);
        }
    }

    async startEvaluation() {
        if (this.selectedModels.size === 0) {
            this.showStatus('请至少选择一个模型', 'warning');
            return;
        }
        
        // Debug: log selected models
        console.log('Selected models:', Array.from(this.selectedModels));

        const config = this.getEvaluationConfig();
        console.log('Evaluation config:', config);
        
        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            this.evaluationId = result.evaluation_id;
            
            this.showStatus('评估已开始', 'success');
            this.startStatusMonitoring();
            this.updateUIForRunningEvaluation();
            
        } catch (error) {
            this.showStatus('启动评估失败: ' + error.message, 'error');
            console.error('Failed to start evaluation:', error);
        }
    }

    async stopEvaluation() {
        try {
            const response = await fetch('/api/evaluation/stop', {
                method: 'POST'
            });
            
            if (response.ok) {
                this.showStatus('评估已停止', 'warning');
                this.stopStatusMonitoring();
                this.updateUIForStoppedEvaluation();
            }
            
        } catch (error) {
            this.showStatus('停止评估失败: ' + error.message, 'error');
            console.error('Failed to stop evaluation:', error);
        }
    }

    startStatusMonitoring() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }
        
        this.statusCheckInterval = setInterval(() => {
            this.checkEvaluationStatus();
        }, 1000);
        
        // Show progress section
        document.getElementById('progress-section').style.display = 'block';
    }

    stopStatusMonitoring() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
    }

    async checkEvaluationStatus() {
        try {
            const response = await fetch('/api/evaluation/status');
            const status = await response.json();
            
            this.updateProgressDisplay(status);
            
            if (status.status === 'completed') {
                this.onEvaluationCompleted();
            } else if (status.status === 'failed') {
                this.onEvaluationFailed(status.error);
            } else if (status.status === 'stopped') {
                this.onEvaluationStopped();
            }
            
        } catch (error) {
            console.error('Failed to check evaluation status:', error);
        }
    }

    updateProgressDisplay(status) {
        const progress = status.progress || {};
        
        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const percentage = progress.percentage || 0;
        progressFill.style.width = `${percentage}%`;
        
        // Update progress text
        document.getElementById('progress-percentage').textContent = `${Math.round(percentage)}%`;
        document.getElementById('progress-message').textContent = progress.message || '处理中...';
        
        // Update progress details
        const detailsElement = document.getElementById('progress-details');
        let detailsHtml = '';
        
        // Show selected models
        if (status.selected_models && status.selected_models.length > 0) {
            detailsHtml += `<strong>评估模型:</strong> ${status.selected_models.join(', ')}<br>`;
        }
        
        if (progress.phase && progress.current !== undefined && progress.total !== undefined) {
            detailsHtml += `
                <strong>当前阶段:</strong> ${this.translatePhase(progress.phase)} <br>
                <strong>进度:</strong> ${progress.current} / ${progress.total}
            `;
        }
        
        detailsElement.innerHTML = detailsHtml;
    }

    translatePhase(phase) {
        const translations = {
            'preparation': '准备阶段',
            'model_': '模型评估: ',
            'completion': '完成阶段'
        };
        
        for (const [key, value] of Object.entries(translations)) {
            if (phase.startsWith(key)) {
                return value + phase.replace(key, '');
            }
        }
        
        return phase;
    }

    async onEvaluationCompleted() {
        this.stopStatusMonitoring();
        this.showStatus('评估完成!', 'success');
        
        try {
            const response = await fetch(`/api/evaluation/results/${this.evaluationId}`);
            this.currentResults = await response.json();
            
            this.displayResults();
            this.updateUIForCompletedEvaluation();
            
        } catch (error) {
            this.showStatus('加载结果失败: ' + error.message, 'error');
            console.error('Failed to load results:', error);
        }
    }

    onEvaluationFailed(error) {
        this.stopStatusMonitoring();
        this.showStatus(`评估失败: ${error}`, 'error');
        this.updateUIForStoppedEvaluation();
    }

    onEvaluationStopped() {
        this.stopStatusMonitoring();
        this.updateUIForStoppedEvaluation();
    }

    displayResults() {
        document.getElementById('results-section').style.display = 'block';
        document.getElementById('export-section').style.display = 'block';
        
        this.displaySummary();
        this.displayDetailedAnalysis();
        this.displayBiasMetrics();
        this.displayErrorAnalysis();
    }

    displaySummary() {
        const results = this.currentResults.results;
        const tbody = document.getElementById('summary-tbody');
        tbody.innerHTML = '';
        
        Object.entries(results).forEach(([modelName, modelResults]) => {
            const row = tbody.insertRow();
            
            row.insertCell(0).textContent = modelName;
            row.insertCell(1).textContent = (modelResults.accuracy * 100).toFixed(1) + '%';
            row.insertCell(2).textContent = modelResults.total_cases.toLocaleString();
            row.insertCell(3).textContent = modelResults.correct_predictions.toLocaleString();
            row.insertCell(4).textContent = modelResults.bias_metrics.grade;
            row.insertCell(5).textContent = modelResults.execution_time.toFixed(1) + 's';
        });
    }

    displayDetailedAnalysis() {
        const container = document.getElementById('detailed-analysis');
        container.innerHTML = '';
        
        const results = this.currentResults.results;
        
        Object.entries(results).forEach(([modelName, modelResults]) => {
            const section = document.createElement('div');
            section.innerHTML = `<h4>${modelName} - 详细分析</h4>`;
            
            // Pronoun analysis
            const pronounTable = this.createAnalysisTable(
                '按代词类型分析',
                modelResults.results_by_pronoun,
                ['代词', '总数', '正确', '准确率']
            );
            
            // 9-category combination analysis (primary)
            const combinationTable = this.createAnalysisTable(
                '按名字-代词组合分析 (9类)',
                modelResults.results_by_combination,
                ['组合类别', '总数', '正确', '准确率', 'A选择率', 'B选择率', 'C选择率']
            );
            
            // Legacy analysis (kept for reference)
            const nameTable = this.createAnalysisTable(
                '按名字类别分析 (传统)',
                modelResults.results_by_name_category,
                ['名字类别', '总数', '正确', '准确率']
            );
            
            const testTable = this.createAnalysisTable(
                '按测试类型分析 (传统)',
                modelResults.results_by_test_type,
                ['测试类型', '总数', '正确', '准确率']
            );
            
            section.appendChild(pronounTable);
            section.appendChild(combinationTable);  // Primary analysis first
            section.appendChild(nameTable);        // Legacy
            section.appendChild(testTable);        // Legacy
            container.appendChild(section);
        });
    }

    createAnalysisTable(title, data, headers) {
        const container = document.createElement('div');
        container.innerHTML = `<h5 style="margin: 20px 0 10px 0;">${title}</h5>`;
        
        const table = document.createElement('table');
        table.className = 'results-table';
        
        // Header
        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        
        // Body
        const tbody = table.createTBody();
        Object.entries(data).forEach(([key, stats]) => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = key;
            row.insertCell(1).textContent = stats.total;
            row.insertCell(2).textContent = stats.correct;
            row.insertCell(3).textContent = (stats.accuracy * 100).toFixed(1) + '%';
            
            // If headers include choice distribution, add those columns
            if (headers.length > 4 && stats.choice_distribution) {
                const total = stats.total;
                row.insertCell(4).textContent = total > 0 ? ((stats.choice_distribution.A || 0) / total * 100).toFixed(1) + '%' : '0%';
                row.insertCell(5).textContent = total > 0 ? ((stats.choice_distribution.B || 0) / total * 100).toFixed(1) + '%' : '0%';
                row.insertCell(6).textContent = total > 0 ? ((stats.choice_distribution.C || 0) / total * 100).toFixed(1) + '%' : '0%';
            }
        });
        
        container.appendChild(table);
        return container;
    }

    displayBiasMetrics() {
        const container = document.getElementById('bias-metrics');
        container.innerHTML = '';
        
        const results = this.currentResults.results;
        const modelName = Object.keys(results)[0];
        
        if (!modelName) return;
        
        const biasMetrics = results[modelName].bias_metrics;
        
        const metrics = [
            { name: '名字依赖度', value: biasMetrics.name_dependency_score, description: '模型被名字影响的程度' },
            { name: '二元固化度', value: biasMetrics.binary_rigidity_score, description: '坚持传统二元分类的程度' },
            { name: '过度确定性', value: biasMetrics.overconfidence_rate, description: '在歧义情况下选择确定答案的比例' },
            { name: '非二元识别率', value: biasMetrics.neo_pronoun_recognition_rate, description: '对新代词的理解程度' },
            { name: 'They理解度', value: biasMetrics.they_comprehension_score, description: '区分单数/复数they的能力' },
            { name: '错位容忍度', value: biasMetrics.mismatch_tolerance_score, description: '基于代词而非名字做判断的能力' }
        ];
        
        metrics.forEach(metric => {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-value">${(metric.value * 100).toFixed(1)}%</div>
                <div class="metric-label">${metric.name}</div>
                <div style="font-size: 11px; color: #999; margin-top: 5px;">${metric.description}</div>
            `;
            container.appendChild(card);
        });
    }

    displayErrorAnalysis() {
        const container = document.getElementById('error-analysis');
        container.innerHTML = '';
        
        const results = this.currentResults.results;
        const modelName = Object.keys(results)[0];
        
        if (!modelName) return;
        
        const errorCases = results[modelName].error_cases.slice(0, 10);
        
        if (errorCases.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: #666;">没有发现错误案例</p>';
            return;
        }
        
        const table = document.createElement('table');
        table.className = 'results-table';
        
        // Header
        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        ['句子', '期望答案', '模型回答', '原始响应'].forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        
        // Body
        const tbody = table.createTBody();
        errorCases.forEach(errorCase => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = errorCase.sentence;
            row.insertCell(1).textContent = errorCase.expected;
            row.insertCell(2).textContent = errorCase.predicted || 'N/A';
            row.insertCell(3).textContent = errorCase.raw_response;
        });
        
        container.appendChild(table);
    }

    switchTab(tabName) {
        // Remove active class from all tabs and content
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        // Add active class to selected tab and content
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    updateUIForRunningEvaluation() {
        document.getElementById('start-evaluation').disabled = true;
        document.getElementById('stop-evaluation').disabled = false;
        
        // Disable all form inputs
        document.querySelectorAll('input, select').forEach(input => {
            input.disabled = true;
        });
    }

    updateUIForStoppedEvaluation() {
        document.getElementById('start-evaluation').disabled = false;
        document.getElementById('stop-evaluation').disabled = true;
        
        // Enable all form inputs
        document.querySelectorAll('input, select').forEach(input => {
            input.disabled = false;
        });
        
        // Hide progress section
        document.getElementById('progress-section').style.display = 'none';
    }

    updateUIForCompletedEvaluation() {
        this.updateUIForStoppedEvaluation();
    }

    exportResults() {
        if (!this.currentResults) {
            this.showStatus('没有可导出的结果', 'warning');
            return;
        }
        
        const dataStr = JSON.stringify(this.currentResults, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `reverse_inference_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        link.click();
        
        this.showStatus('结果已导出', 'success');
    }

    saveConfiguration() {
        const config = this.getEvaluationConfig();
        // Remove API keys from config for security
        delete config.api_keys;
        
        const dataStr = JSON.stringify(config, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `reverse_inference_config_${new Date().toISOString().slice(0, 10)}.json`;
        link.click();
        
        this.showStatus('配置已保存', 'success');
    }

    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('status-message');
        statusElement.textContent = message;
        statusElement.className = `status-message status-${type}`;
        statusElement.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 5000);
    }

    // Model-specific parameter suggestions
    updateParameterSuggestions() {
        const selectedModels = Array.from(this.selectedModels);
        const hasGPTModel = selectedModels.some(model => model.toLowerCase().includes('gpt'));
        const hasAnthropicModel = selectedModels.some(model => model.toLowerCase().includes('claude'));
        
        // Get current values
        const batchSizeInput = document.getElementById('batch-size');
        const maxConcurrentInput = document.getElementById('max-concurrent');
        const timeoutInput = document.getElementById('timeout');
        
        if (hasGPTModel) {
            // GPT models need conservative settings due to rate limits
            this.showParameterSuggestion('GPT模型检测到，建议使用保守设置以避免速率限制');
            
            // Only suggest if current values are higher than recommended
            if (parseInt(batchSizeInput.value) > 5) {
                batchSizeInput.value = '1';
                batchSizeInput.style.backgroundColor = '#fff3cd';
            }
            if (parseInt(maxConcurrentInput.value) > 3) {
                maxConcurrentInput.value = '1';
                maxConcurrentInput.style.backgroundColor = '#fff3cd';
            }
            if (parseInt(timeoutInput.value) < 60) {
                timeoutInput.value = '120';
                timeoutInput.style.backgroundColor = '#fff3cd';
            }
        } else if (hasAnthropicModel) {
            // Anthropic models can handle moderate settings
            this.showParameterSuggestion('Claude模型检测到，可以使用中等并发设置');
            
            if (parseInt(batchSizeInput.value) < 10) {
                batchSizeInput.value = '10';
                batchSizeInput.style.backgroundColor = '#d1ecf1';
            }
            if (parseInt(maxConcurrentInput.value) < 3) {
                maxConcurrentInput.value = '3';
                maxConcurrentInput.style.backgroundColor = '#d1ecf1';
            }
        } else {
            // Other models can use higher settings
            this.showParameterSuggestion('其他模型可以使用较高的并发设置');
            
            if (parseInt(batchSizeInput.value) < 20) {
                batchSizeInput.value = '20';
                batchSizeInput.style.backgroundColor = '#d4edda';
            }
            if (parseInt(maxConcurrentInput.value) < 5) {
                maxConcurrentInput.value = '5';
                maxConcurrentInput.style.backgroundColor = '#d4edda';
            }
        }
        
        // Clear background color after 3 seconds
        setTimeout(() => {
            batchSizeInput.style.backgroundColor = '';
            maxConcurrentInput.style.backgroundColor = '';
            timeoutInput.style.backgroundColor = '';
        }, 3000);
    }
    
    showParameterSuggestion(message) {
        // Create or update suggestion message
        let suggestionDiv = document.getElementById('parameter-suggestion');
        if (!suggestionDiv) {
            suggestionDiv = document.createElement('div');
            suggestionDiv.id = 'parameter-suggestion';
            suggestionDiv.style.cssText = `
                background: #e3f2fd;
                color: #1565c0;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #2196f3;
                font-size: 14px;
            `;
            
            const configSection = document.querySelector('.section h3').parentNode;
            configSection.insertBefore(suggestionDiv, configSection.children[1]);
        }
        
        suggestionDiv.innerHTML = `💡 ${message}`;
        suggestionDiv.style.display = 'block';
        
        // Hide after 5 seconds
        setTimeout(() => {
            suggestionDiv.style.display = 'none';
        }, 5000);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.reverseInferenceApp = new ReverseInferenceApp();
});