// analysis_full.cpp
// --------------------------------------------------------
// Scalable C++ Analysis for Large-Scale Multi-Agent Experiments
// Utilizes matplotlib-cpp & Eigen for high-performance computation and plotting
// Author: Laksh Patel
// Date: 2025-06-17

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>
#include "matplotlibcpp.h"  // Requires matplotlib-cpp installed

namespace plt = matplotlibcpp;

// Utility: read CSV column into vector<T>
template<typename T>
std::vector<T> readColumn(const std::string &file, const std::string &colName) {
    std::ifstream in(file);
    std::string header;
    getline(in, header);
    std::vector<std::string> cols;
    std::stringstream hs(header);
    std::string name;
    while (getline(hs, name, ',')) cols.push_back(name);
    int idx = std::find(cols.begin(), cols.end(), colName) - cols.begin();
    std::vector<T> data;
    std::string line;
    while (getline(in, line)) {
        std::stringstream ls(line);
        std::string field;
        for (int i = 0; i <= idx; ++i) getline(ls, field, ',');
        data.push_back(static_cast<T>(std::stod(field)));
    }
    return data;
}

// Read aggregated metrics (means+std) into Eigen matrix
Eigen::MatrixXd loadAggregated(const std::string &path, std::vector<std::string> &methods) {
    std::ifstream in(path);
    std::string line;
    getline(in, line);
    std::stringstream hs(line);
    std::vector<std::string> headers;
    std::string h;
    while (getline(hs, h, ',')) headers.push_back(h);
    std::vector<std::vector<double>> rows;
    while (getline(in, line)) {
        std::stringstream ls(line);
        std::string field;
        // method name
        getline(ls, field, ','); methods.push_back(field);
        std::vector<double> vals;
        while (getline(ls, field, ',')) vals.push_back(stod(field));
        rows.push_back(vals);
    }
    int m = rows.size(), n = rows[0].size();
    Eigen::MatrixXd M(m,n);
    for (int i=0;i<m;++i)
        for (int j=0;j<n;++j)
            M(i,j) = rows[i][j];
    return M;
}

// Load error curves per method & agent from CSV: "results/error_curves_{method}_agent{agent}.csv"
std::map<std::string, std::map<int, std::vector<double>>> loadAgentErrorCurves(
    const std::vector<std::string> &methods, const std::string &dir, int maxAgents) {
    std::map<std::string, std::map<int, std::vector<double>>> data;
    for (auto &m: methods) {
        for (int a=0; a<maxAgents; ++a) {
            std::string fname = dir + "/error_curves_" + m + "_agent" + std::to_string(a) + ".csv";
            std::ifstream in(fname);
            if (!in.good()) continue;
            std::string line;
            std::vector<double> curve;
            while (getline(in, line)) curve.push_back(stod(line));
            data[m][a] = curve;
        }
    }
    return data;
}

int main() {
    // Load methods and aggregated metrics
    std::vector<std::string> methods;
    Eigen::MatrixXd agg = loadAggregated("results/aggregated_metrics.csv", methods);
    int numMethods = methods.size();

    // Determine number of agents from trial_residuals.csv
    auto agents = readColumn<int>("results/trial_residuals.csv","Agent");
    int maxAgent = *max_element(agents.begin(), agents.end()) + 1;
    std::cout<<"Detected "<<maxAgent<<" agents.\n";

    // 1) Plot aggregated V100 and V_inf for each agent across methods
    for (int a=0; a<maxAgent; ++a) {
        std::vector<double> V100_agent;
        std::vector<double> Vinf_agent;
        for (int i=0; i<numMethods; ++i) {
            // Column indices: 0=V100_mean, 2=V_inf_mean
            V100_agent.push_back(agg(i,0));
            Vinf_agent.push_back(agg(i,2));
        }
        plt::figure();
        plt::bar(methods, V100_agent, {"alpha",0.6});
        plt::title("Agent " + std::to_string(a) + " - V100");
        plt::savefig("results/figures/agent"+std::to_string(a)+"_V100.png");
        plt::clf();

        plt::bar(methods, Vinf_agent, {"color","orange"});
        plt::title("Agent " + std::to_string(a) + " - V_inf");
        plt::savefig("results/figures/agent"+std::to_string(a)+"_Vinf.png");
        plt::clf();
    }

    // 2) Load and plot error curves per agent
    auto agentCurves = loadAgentErrorCurves(methods, "results", maxAgent);
    for (auto &m: methods) {
        if (!agentCurves.count(m)) continue;
        plt::figure();
        for (auto &ac: agentCurves[m]) {
            plt::plot(ac.second, {"label","Agent"+std::to_string(ac.first)});
        }
        plt::legend();
        plt::title(""+m+" Error Curves by Agent");
        plt::savefig("results/figures/"+m+"_by_agent.png");
        plt::clf();
    }

    // 3) Heatmap: residual over time & agent for each method
    auto residuals = readColumn<double>("results/trial_residuals.csv","Residual");
    auto meths = readColumn<std::string>("results/trial_residuals.csv","Method");
    auto times = readColumn<int>("results/trial_residuals.csv","TimeStep");
    std::map<std::string, Eigen::MatrixXd> heatmaps;
    for (int i=0;i<residuals.size();++i) {
        int a = agents[i], t = times[i];
        std::string m = meths[i];
        if (!heatmaps.count(m)) heatmaps[m] = Eigen::MatrixXd::Zero(maxAgent, agg.cols());
        heatmaps[m](a, t) = residuals[i];
    }
    for (auto &hm: heatmaps) {
        auto &M = hm.second;
        // Convert to vector of vectors
        std::vector<std::vector<double>> mat(maxAgent);
        for (int r=0;r<maxAgent;++r) {
            mat[r] = std::vector<double>(M.row(r).data(), M.row(r).data()+M.cols());
        }
        plt::figure();
        plt::imshow(mat);
        plt::colorbar();
        plt::title(hm.first + " Residual Heatmap");
        plt::xlabel("Time Step"); plt::ylabel("Agent");
        plt::savefig("results/figures/heatmap_"+hm.first+".png");
        plt::clf();
    }

    // 4) Global PCA across all agents & methods
    // Flatten data: each agent-method as sample, features = time-series points
    std::vector<std::vector<double>> samples;
    std::vector<std::string> labels;
    for (auto &m: methods) {
        for (auto &ac: agentCurves[m]) {
            samples.push_back(ac.second);
            labels.push_back(m+"_A"+std::to_string(ac.first));
        }
    }
    int S = samples.size(), T = samples[0].size();
    Eigen::MatrixXd X(S, T);
    for (int i=0;i<S;++i)
        for (int j=0;j<T;++j)
            X(i,j) = samples[i][j];
    // PCA via SVD
    Eigen::RowVectorXd mu = X.colwise().mean();
    Eigen::MatrixXd X0 = X.rowwise() - mu;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X0, Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    // Plot first 2 PCs
    plt::figure();
    for (int i=0;i<S;++i) {
        plt::scatter(U(i,0), U(i,1));
        plt::text(U(i,0),U(i,1), labels[i]);
    }
    plt::title("PCA: Agent-Method Samples");
    plt::savefig("results/figures/pca_agents.png");

    std::cout<<"Large-scale agent analysis complete.\n";
    return 0;
}
