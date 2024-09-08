#include <string>  
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>  
#include <algorithm> 
#include <utility>
#include <cmath>

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

xy_values openCSV() { // return a struct, that is why we use xy_values before declaring a function
    xy_values values;
    std::ifstream file("C:/Users/janar/Desktop/cplusplus/data.csv");  
    if (!file.is_open()) {
        std::cout << "Error opening file!" << std::endl;
    };

    // std::string data[1000][1000];
    std::vector<std::vector<double>> photometric_data; // two dimesional vector, vectors because auto scaling
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
                    // std::cout << current_index << '\n';
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
        double leaf_node_value;

        Node() : feature_index(-1), threshold(0), left(nullptr), right(nullptr), var_red(0), leaf_node_value(0) {}
    
        // Constructor for internal nodes
        Node(int feature_index, double threshold, Node* left, Node* right, double var_red) 
            : feature_index(feature_index), threshold(threshold), left(left), right(right), var_red(var_red) {}

        // Constructor for leaf nodes
        Node(double leaf_node_value) : feature_index(-1), threshold(0), left(nullptr), right(nullptr), var_red(0), leaf_node_value(leaf_node_value) {}
};

class DecisionTree {
    public:
        std::vector<std::vector<double>> root;
        double min_samples_split = 2;
        int max_depth = 5;

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
                int threshold;
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
                // std::cout << possible_thresholds_sliced.size();
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
                    // split(photometric_vector_data, z_vector_data, feature_index, threshold);
                    // std::cout << threshold;
                }

            }

            return best_split;
        }

        Node* build_tree(std::pair<std::vector<std::vector<double>>, std::vector<double>>& dataset, int current_depth = 0) {
            // xy_values dataset = openCSV();
            auto [x, y] = dataset;

            
            // int num_samples = x.size();
            // int num_features = x[0].size();
            int num_samples = 50;
            int num_features = 9;

            // std::cout << dataset->redshift_vector.size();

            if (num_samples >= min_samples_split && current_depth <= max_depth) {
                auto best_split = get_best_split(dataset, num_samples, num_features);
                std::cout << best_split.var_red;
                // std::cout << best_split.dataset_left.second.size();

                if (best_split.var_red > 0) {
                    Node* left_subtree = build_tree(best_split.dataset_left, current_depth + 1);
                    Node* right_subtree = build_tree(best_split.dataset_right, current_depth + 1);

                    // Node node;
                    // node.feature_index = best_split.feature_index;
                    // node.threshold = best_split.threshold;
                    // node.left = left_subtree;
                    // node.right = right_subtree;
                    // node.var_red = best_split.var_red;
                    // return node;
                    // return Node(best_split.feature_index, best_split.threshold);
                    return new Node(best_split.feature_index, best_split.threshold, left_subtree, right_subtree, best_split.var_red);
                }
            }
            double leaf_value = calculate_leaf_value(y);
            std::cout << leaf_value;
            return new Node(leaf_value);
            // double leaf_value = calculate_leaf_value(y);
            // return std::make_unique<Node>(leaf_value);
        } 
        
};

// class RandomForestRegressor {
//     int n_trees = 20;
//     int min_samples_split = 2;
//     int max_depth = 2;
//     std::vector<>
// };




int main() {
    xy_values dataset = openCSV();
    // auto *dataset_pointer = &dataset;
    DecisionTree DecisionTree;

    // std::vector<std::vector<double>> photometric_data_sliced (x.begin() + 1, x.begin() + 51);
    // std::vector<double> z_data_sliced (y.begin() + 1, y.begin() + 51); 

    // auto& [x, y] = dataset;
    // std::pair<std::vector<std::vector<double>>, std::vector<double>> data = { dataset.photometric_vector, dataset.redshift_vector };
    // {(dataset.photometric_vector.begin() + 1, dataset.photometric_vector.begin() + 51), (dataset.redshift_vector.begin() + 1, dataset.redshift_vector.begin() + 51)}

    auto photometric = std::vector<std::vector<double>> (dataset.photometric_vector.begin() + 1, dataset.photometric_vector.begin() + 51);
    auto redshift = std::vector<double> (dataset.redshift_vector.begin() + 1, dataset.redshift_vector.begin() + 51);

    std::pair<std::vector<std::vector<double>>, std::vector<double>> data = { photometric, redshift };
    DecisionTree.build_tree(data);

    return 0;
}
