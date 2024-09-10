#include <string>  
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>  
#include <algorithm> 
#include <utility>
#include <cmath>
#include <random>

// hold photometric and redshift data
struct xy_values {
    std::vector<std::vector<double>> photometric_vector;
    std::vector<double> redshift_vector;
};

double stringToDouble(std::string& str) {
    try {
        return std::stod(str);  // std::stod converts string to double
    } catch (std::string str) {
        std::cout << "a fucking string";
        return 0.0;
    }
}

// return a struct, containing data for processing
xy_values openCSV() { // return a struct, that is why we use xy_values before declaring a function
    xy_values values;
    std::ifstream file("C:/Users/janar/Desktop/cplusplus/data.csv");  
    if (!file.is_open()) {
        std::cout << "Error opening file!" << std::endl;
    };

    // std::string data[1000][1000];
    std::vector<std::vector<double>> photometric_data; 
    std::vector<double> redshift_data;
    std::string line;
    
    std::getline(file, line);
    while (std::getline(file, line)) { // getline reads a line, if not in use the spaces between data will mean new input, i think?
        std::stringstream obj(line); // object like string
        std::string cell;
        std::vector<double> row;
        int current_index = 0;

        while (getline(obj, cell, ',')) {

                double value = stringToDouble(cell);

                if (current_index >= 4 && current_index < 13) {
                    row.push_back(value);
                }

                if (current_index == 13) {
                    redshift_data.push_back(value);
                }

            current_index++;
        }

        photometric_data.push_back(row);
    }

    file.close();

    values.photometric_vector = photometric_data;
    values.redshift_vector = redshift_data;

    return values;
}

class Node {
    public:
        int feature_index;
        double threshold;
        Node* left;
        Node* right;
        double var_red;
        double leaf_node_value = 0;

        Node() : feature_index(-1), threshold(0), left(nullptr), right(nullptr), var_red(0), leaf_node_value(0) {}
    
        // Constructor for internal nodes
        Node(int feature_index, double threshold, Node* left, Node* right, double var_red) 
            : feature_index(feature_index), threshold(threshold), left(left), right(right), var_red(var_red) {}

        // Constructor for leaf nodes
        Node(double leaf_node_value) : feature_index(-1), threshold(0), left(nullptr), right(nullptr), var_red(0), leaf_node_value(leaf_node_value) {}
};

class DecisionTree {
    public:
        Node* root;
        double min_samples_split = 2;
        int max_depth = 10;

        double calculate_leaf_value(std::vector<double> y) {
            double summary = 0;
            for (double num : y) {
                summary += num;
            }

            return summary / y.size();
        }

        double var(std::vector<double> element) {
            double modified_sum = 0;
            double sum = 0;
            for (int x = 0; x < element.size(); x++) { sum += element[x]; };
            double mean_value = sum / element.size();

            for (int x = 0; x < element.size(); x++) { modified_sum += pow((element[x] - mean_value), 2); };

            return modified_sum / element.size();
        }

        double variance_reduction(std::vector<double> z_vector, std::vector<double> z_vector_left, std::vector<double> z_vector_right) {
            double weight_l = static_cast<double>(z_vector_left.size()) / z_vector.size();
            double weight_r = static_cast<double>(z_vector_right.size()) / z_vector.size();
            double reduction = var(z_vector) - (weight_l * var(z_vector_left) + weight_r * var(z_vector_right));
            return reduction;
        }

        std::pair<std::pair<std::vector<std::vector<double>>, std::vector<double>>, std::pair<std::vector<std::vector<double>>, std::vector<double>>> 
        split(std::vector<std::vector<double>> photometric, std::vector<double> redshift, int feature_index, double threshold) {

            std::vector<std::vector<double>> dataset_left_photo;
            std::vector<double> dataset_left_z;
            std::vector<std::vector<double>> dataset_right_photo;
            std::vector<double> dataset_right_z;

            for (int i = 0; i < photometric.size(); i++) {
                if (photometric.size() != redshift.size()) {
                    std::cout << "data size does not match.";
                }
                std::vector<double> value_photo = photometric[i];
                double value_z = redshift[i];

                if (value_photo[feature_index] <= threshold) {
                    dataset_left_photo.push_back(value_photo);
                    dataset_left_z.push_back(value_z);
                } else {
                    dataset_right_photo.push_back(value_photo);
                    dataset_right_z.push_back(value_z);
                }

            }

            return { {dataset_left_photo, dataset_left_z}, {dataset_right_photo, dataset_right_z} };
        }
        
        auto get_best_split(std::pair<std::vector<std::vector<double>>, std::vector<double>>& dataset, int num_samples, int num_features) {
            
            auto& [photometric_vector_data, z_vector_data] = dataset;
            
            struct best_split {
                int feature_index;
                double threshold;
                std::pair<std::vector<std::vector<double>>, std::vector<double>> dataset_left;
                std::pair<std::vector<std::vector<double>>, std::vector<double>> dataset_right;
                double var_red;
            } best_split;
            double max_var_red = -1;
            auto feature_values = photometric_vector_data;
            std::vector<double> possible_thresholds;

            for (int feature_index = 0; feature_index < num_features; feature_index++) {
                
                std::vector<double> feature_column;
                for (std::vector<double> x : feature_values) {
                    feature_column.push_back(x[feature_index]);
                }

                for (size_t val = 0; val < feature_column.size(); val++) {
                    if (std::find(possible_thresholds.begin(), possible_thresholds.end(), feature_column[val]) == possible_thresholds.end()) {
                        possible_thresholds.push_back(feature_column[val]);
                    }
                }
                // sort 
                std::sort(possible_thresholds.begin(), possible_thresholds.end());
                for (double threshold : possible_thresholds) {
                    auto [dataset_left, dataset_right] = split(photometric_vector_data, z_vector_data, feature_index, threshold);

                    if (dataset_left.first.size() > 0 && dataset_right.first.size() > 0) {
                        double curr_var_red = variance_reduction(z_vector_data, dataset_left.second, dataset_right.second);
                        
                        if (curr_var_red > max_var_red) {
                            best_split.feature_index = feature_index;
                            best_split.threshold = threshold;
                            best_split.dataset_left = dataset_left;
                            best_split.dataset_right = dataset_right;
                            best_split.var_red = curr_var_red;
                            max_var_red = curr_var_red;
                        }
                    }
                }

            }
            return best_split;
        }

        Node* build_tree(std::pair<std::vector<std::vector<double>>, std::vector<double>>& dataset, int current_depth = 0) {
            auto [x, y] = dataset;

            int num_samples = static_cast<int> (x.size());
            int num_features = static_cast<int> (x[0].size());

            if (num_samples >= min_samples_split && current_depth <= max_depth) {
                auto best_split = get_best_split(dataset, num_samples, num_features); // get the best split

                if (best_split.var_red > 0) {
                    // recursion for left and right subtree
                    Node* left_subtree = build_tree(best_split.dataset_left, current_depth + 1);
                    Node* right_subtree = build_tree(best_split.dataset_right, current_depth + 1);

                    return new Node(best_split.feature_index, best_split.threshold, left_subtree, right_subtree, best_split.var_red);
                }
            }
            double leaf_value = calculate_leaf_value(y); // calculate the mean of a leaf
            
            return new Node(leaf_value); // Node 'object'
        } 

        // singular tree fitting
        void fit(std::pair<std::vector<std::vector<double>>, std::vector<double>>& dataset) {
            DecisionTree DecisionTree;
            root = DecisionTree.build_tree(dataset);
        }

        // predict a value, singular tree
        auto make_prediction(std::vector<double> x, Node* tree) {
            if (tree->leaf_node_value) {
                return tree->leaf_node_value;
            }
            auto feature_val = x[tree->feature_index];

            if (feature_val <= tree->threshold) {
                return make_prediction(x, tree->left);
            } else {
                return make_prediction(x, tree->right);
            }
        }

        auto predict(std::vector<std::vector<double>> x) {
            std::vector<double> predictions;
            for (int i = 0; i < x.size(); i++) {
                double value_predicted = make_prediction(x[i], root);
                predictions.push_back(value_predicted);
            }

            return predictions;
        }
        
};

// for random forest
class RandomForestRegressor {
    public:
        int n_trees = 20;
        int min_samples_split = 2;
        // int max_depth = 2;
        std::vector<DecisionTree> trees;

    std::pair<std::vector<std::vector<double>>, std::vector<double>> 
    bootstrap(std::pair<std::vector<std::vector<double>>, std::vector<double>>& dataset) {
        std::vector<int> bootstrap_indices;
        auto [x, y] = dataset;
        int n_samples = static_cast<int> (x.size());

        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_int_distribution<> distr(0, n_samples - 1); // define the range

        std::vector<std::vector<double>> x_bootstrap; // photometric data
        std::vector<double> y_bootstrap; // redshift data

        for (int num = 0; num < n_samples; num++) {
            // bootstrap_indices.push_back(distr(gen));
            int rand = distr(gen);
            x_bootstrap.push_back(x[rand]);
            y_bootstrap.push_back(y[rand]);
        }

        return { x_bootstrap, y_bootstrap };
    }

    void fit(std::pair<std::vector<std::vector<double>>, std::vector<double>>& dataset) {
        trees.clear();

        for (int i = 0; i < n_trees; i++) {
            auto [x, y] = bootstrap(dataset);
            DecisionTree DecisionTree;
            DecisionTree.fit(dataset);
            trees.push_back(DecisionTree);
        }
    }

    auto predict(std::vector<std::vector<double>> x) {
        std::vector<std::vector<double>> output_from_trees(x.size(), std::vector<double>(trees.size(), 0.0));
        std::vector<double> mean_values;
        for (int i = 0; i < trees.size(); i++) {
            std::vector<double> predictions = trees[i].predict(x);
            
            // output vector
            for (int j = 0; j < predictions.size(); j++) {
                output_from_trees[j][i] = predictions[j];
            }
        }

        for (int i = 0; i < output_from_trees.size(); i++) {
            double sum = 0;
            for (int ii = 0; ii < output_from_trees[i].size(); ii++) {
                sum += output_from_trees[i][ii];
            }

            double mean = sum / output_from_trees[i].size();
            mean_values.push_back(mean);
        }
        return mean_values;
    }
};


int main() {
    xy_values dataset = openCSV();
    RandomForestRegressor RandomForestRegressor;

    auto photometric = std::vector<std::vector<double>> (dataset.photometric_vector.begin(), dataset.photometric_vector.begin() + 194960);
    auto redshift = std::vector<double> (dataset.redshift_vector.begin(), dataset.redshift_vector.begin() + 194960);

    auto photometric_test = std::vector<std::vector<double>> (dataset.photometric_vector.begin() + 194960, dataset.photometric_vector.end());
    auto redshift_test = std::vector<double> (dataset.redshift_vector.begin() + 194960, dataset.redshift_vector.end());

    std::pair<std::vector<std::vector<double>>, std::vector<double>> data = { photometric, redshift };

    RandomForestRegressor.fit(data);

    auto y_pred = RandomForestRegressor.predict(photometric_test);

    double sum = 0;
    for (int i = 0; i < y_pred.size(); i++) {
        sum += abs(redshift_test[i] - y_pred[i]);
    }

    double size = static_cast<double> (redshift_test.size());
    std::cout << sum / size;

    return 0;
}