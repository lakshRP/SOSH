% analysis_extensive.m
% --------------------------------------------------------
% Comprehensive MATLAB Analysis for SOSH Experiment Results
% Loads datasets, computes advanced statistics, generates extensive plots,
% performs hypothesis testing, clustering, regression, PCA, and exports summaries.

% Author: Laksh Patel
% Date: 2025-06-17

%% 1) Setup
clearvars; close all; clc;
addpath(genpath(pwd));  % Ensure all subfolders are on path
resultsDir = 'results';  % Directory containing CSV, NPZ/Mat files
figDir = fullfile(resultsDir, 'figures');
if ~exist(figDir,'dir'), mkdir(figDir); end

disp('Starting extensive MATLAB analysis...');

%% 2) Load Aggregated Metrics
metricsFile = fullfile(resultsDir, 'aggregated_metrics.csv');
opts = detectImportOptions(metricsFile);
metricsTbl = readtable(metricsFile, opts);
methods = metricsTbl.Method;
numMethods = height(metricsTbl);

%% 3) Load Error Curves
% Assume error_curves.mat exists, containing variable "errorCurves" as struct with fields per method
matFile = fullfile(resultsDir, 'error_curves.mat');
if exist(matFile,'file')
    S = load(matFile);
    errorCurves = S.errorCurves;
else
    % Fallback: use Python to load NPZ
    npzFile = fullfile(resultsDir, 'error_curves.npz');
    if exist(npzFile,'file')
        pyData = py.numpy.load(npzFile);
        pyMethods = cellfun(@char, cell(pyData.files), 'UniformOutput', false);
        for i=1:length(pyMethods)
            errorCurves.(pyMethods{i}) = double(pyData.get(pyMethods{i}));
        end
    else
        errorCurves = struct();
        warning('No error curves file found.');
    end
end

%% 4) Load All Positions
posFile = fullfile(resultsDir, 'all_positions.csv');
posTbl = readtable(posFile);

%% 5) Basic Summary Statistics
summaryStats = table();
summaryStats.Method = methods;
summaryStats.V100_mean = metricsTbl.V100_mean;
summaryStats.V100_std  = metricsTbl.V100_std;
summaryStats.Vinf_mean = metricsTbl.V_inf_mean;
summaryStats.Vinf_std  = metricsTbl.V_inf_std;
summaryStats.AUC_mean  = metricsTbl.AUC0_100;
summaryStats.AUC_std   = metricsTbl.AUC_std;
summaryStats.T1_mean   = metricsTbl.T1_mean;
summaryStats.T1_std    = metricsTbl.T1_std;

disp('Summary statistics table:');
disp(summaryStats);

%% 6) Boxplots of Trial Metrics
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
tiledlayout(2,2,'TileSpacing','compact');
metrics_raw = readtable(fullfile(resultsDir,'trial_metrics.csv')); % if exists
if exist('trial_metrics.csv','file')
    nexttile; boxplot(metrics_raw.V100, metrics_raw.Method);
    title('Boxplot V100 by Method'); ylabel('V100');
    nexttile; boxplot(metrics_raw.V_inf, metrics_raw.Method);
    title('Boxplot V_{inf} by Method'); ylabel('V_{inf}');
    nexttile; boxplot(metrics_raw.AUC, metrics_raw.Method);
    title('Boxplot AUC by Method'); ylabel('AUC');
    nexttile; boxplot(metrics_raw.T1, metrics_raw.Method);
    title('Boxplot T1% by Method'); ylabel('T1');
    saveas(gcf, fullfile(figDir,'boxplots_metrics.png'));
end

%% 7) ANOVA: Compare V100 Across Methods
pV100 = anova1(metricsTbl.V100_mean, methods, 'off');
disp(['ANOVA p-value for V100 means: ', num2str(pV100)]);

%% 8) Pairwise Tukey-Kramer Post-hoc for V100
[c,m,h,nms] = multcompare(stats.anova1(metricsTbl.V100_mean, methods), 'Display','off');
writetable(cell2table(nms), fullfile(resultsDir,'v100_tukey_modes.csv'));

%% 9) Correlation Matrix of Metrics
dataMatrix = [metricsTbl.V100_mean, metricsTbl.V_inf_mean, metricsTbl.AUC0_100, metricsTbl.T1_mean];
[R,P] = corrcoef(dataMatrix);
figure; heatmap({'V100','Vinf','AUC','T1'}, {'V100','Vinf','AUC','T1'}, R, 'Colormap', parula);
title('Correlation Matrix of Metrics');
saveas(gcf, fullfile(figDir,'corr_matrix.png'));

%% 10) Principal Component Analysis
[coeff,score,latent,tsquared,explained] = pca(dataMatrix);
figure;
pareto(explained);
title('PCA Explained Variance');
saveas(gcf, fullfile(figDir,'pca_variance.png'));

%% 11) K-means Clustering of Methods
numClusters = 2;
[idx,C] = kmeans(dataMatrix, numClusters, 'Replicates',10);
figure; gscatter(score(:,1), score(:,2), idx);
text(score(:,1), score(:,2), methods, 'VerticalAlignment','bottom','FontSize',8);
title('K-means Clustering on PCA Scores');
xlabel('PC1'); ylabel('PC2');
saveas(gcf, fullfile(figDir,'kmeans_pca.png'));

%% 12) Trajectory Clustering of Error Curves
methodNames = fieldnames(errorCurves);
maxLen = max(cellfun(@(f) length(errorCurves.(f)), methodNames));
curveMatrix = zeros(numMethods, maxLen);
for i=1:numMethods
    c = errorCurves.(methodNames{i});
    curveMatrix(i,1:length(c)) = c;
end

distMat = pdist(curveMatrix, 'euclidean');
Z = linkage(distMat, 'ward');
figure; dendrogram(Z,0,'Labels',methodNames);
title('Dendrogram of Error Curve Similarity');
saveas(gcf, fullfile(figDir,'dendrogram_error_curves.png'));

%% 13) Temporal Heatmap of Formation Error
figure;
imagesc(curveMatrix);
colorbar;
yticks(1:numMethods); yticklabels(methodNames);
xticks(1:20:maxLen);
title('Heatmap of Error Curves Over Time');
xlabel('Time Step'); ylabel('Method');
saveas(gcf, fullfile(figDir,'heatmap_error_curves.png'));

%% 14) Export LaTeX Table of Summary Stats
latexFile = fullfile(resultsDir,'summary_table.tex');
fid = fopen(latexFile,'w');
fprintf(fid,'\\begin{tabular}{lrrrrrrrr}\\n');
fprintf(fid,'Method & V100 & Std & V_{inf} & Std & AUC & Std & T1 & Std \\\\n');
for i=1:numMethods
    fprintf(fid,'%s & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.2f & %.2f \\\\n', ...
        methods{i}, summaryStats.V100_mean(i), summaryStats.V100_std(i), ...
        summaryStats.Vinf_mean(i), summaryStats.Vinf_std(i), ...
        summaryStats.AUC_mean(i), summaryStats.AUC_std(i), ...
        summaryStats.T1_mean(i), summaryStats.T1_std(i));
end
fprintf(fid,'\\end{tabular}\\n');
fclose(fid);
disp(['LaTeX summary table exported to ', latexFile]);

%% 15) Advanced: Fit Exponential Decay to Error Curves
decayParams = zeros(numMethods,2);
for i=1:numMethods
    t = (0:length(curveMatrix(i,:))-1)';
    y = curveMatrix(i,:)';
    ft = fit(t, y, 'a*exp(-b*x) + c', 'StartPoint',[y(1),0.1,y(end)]);
    decayParams(i,:) = [ft.a, ft.b];
end
figure; bar(decayParams(:,2)); xticklabels(methods);
title('Estimated Decay Rates b for Each Method');
ylabel('Decay Rate b');
saveas(gcf, fullfile(figDir,'decay_rates.png'));

%% Done
disp('Extensive MATLAB analysis complete. Figures saved in figures directory.');
