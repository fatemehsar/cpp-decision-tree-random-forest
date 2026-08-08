#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>

// Storing one data sample with its features and matching class
struct Sample {
    int features[10];
    int label;
};

// Showing a leaf or rule node in the decision tree
struct Node {
    bool isLeaf;
    int predictedClass;
    int splitFeature;
    Node* left;
    Node* right;
    std::vector<Sample> data;

    Node() {
        isLeaf = true;
        predictedClass = -1;
        splitFeature = -1;
        left = nullptr;
        right = nullptr;
    }
};

// to be used in priority queue to choose the next leaf that should be split
struct QueueItem {
    double score;
    int feature;
    Node* node;

    bool operator<(const QueueItem& other) const {
        return score > other.score;
    }
};

struct SplitCandidate {
    int feature;
    double score;
};

void countClasses(const std::vector<Sample>& data) {
    int class0 = 0;
    int class1 = 0;
    int class2 = 0;

    for (int i = 0; i < data.size(); i++) {
        if (data[i].label == 0) {
            class0++;
        } else if (data[i].label == 1) {
            class1++;
        } else if (data[i].label == 2) {
            class2++;
        }
    }

    std::cout << "Class 0: " << class0 << std::endl;
    std::cout << "Class 1: " << class1 << std::endl;
    std::cout << "Class 2: " << class2 << std::endl;
}

// calculating the Gini impurity for a sample group
double calculateGini(const std::vector<Sample>& data) {
    if (data.size() == 0) {
        return 0.0;
    }

    int class0 = 0;
    int class1 = 0;
    int class2 = 0;

    for (int i = 0; i < data.size(); i++) {
        if (data[i].label == 0) {
            class0++;
        } else if (data[i].label == 1) {
            class1++;
        } else if (data[i].label == 2) {
            class2++;
        }
    }

    double total = data.size();

    double p0 = class0 / total;
    double p1 = class1 / total;
    double p2 = class2 / total;

    double gini = 1.0 - (p0 * p0 + p1 * p1 + p2 * p2);

    return gini;
}

// dividing data based on one binary feature into left and right groups
void splitDataByFeature(
    const std::vector<Sample>& data,
    int featureIndex,
    std::vector<Sample>& leftData,
    std::vector<Sample>& rightData
) {
    leftData.clear();
    rightData.clear();

    for (int i = 0; i < data.size(); i++) {
        if (data[i].features[featureIndex] == 0) {
            leftData.push_back(data[i]);
        } else {
            rightData.push_back(data[i]);
        }
    }
}

// calculating the improvement of impurity after splitting on a feature
double calculateSplitScore(const std::vector<Sample>& data, int featureIndex) {
    std::vector<Sample> leftData;
    std::vector<Sample> rightData;

    splitDataByFeature(data, featureIndex, leftData, rightData);

    if (leftData.size() == 0 || rightData.size() == 0) {
        return 0.0;
    }

    double parentGini = calculateGini(data);
    double leftGini = calculateGini(leftData);
    double rightGini = calculateGini(rightData);

    double totalSize = data.size();

    double weightedChildGini =
        (leftData.size() / totalSize) * leftGini +
        (rightData.size() / totalSize) * rightGini;

    double improvement = (weightedChildGini - parentGini) * totalSize;

    return improvement;
}

// finding the best feature to split on that feature, which is chosen from feature list
SplitCandidate findBestSplit(const std::vector<Sample>& data, const std::vector<int>& allowedFeatures) {
    SplitCandidate best;
    best.feature = -1;
    best.score = 0.0;

    for (int i = 0; i < allowedFeatures.size(); i++) {
        int feature = allowedFeatures[i];
        double score = calculateSplitScore(data, feature);

        if (score < best.score) {
            best.score = score;
            best.feature = feature;
        }
    }

    return best;
}

int findBestFeature(const std::vector<Sample>& data, const std::vector<int>& allowedFeatures) {
    return findBestSplit(data, allowedFeatures).feature;
}

// finding the most common class in a sample group
int majorityClass(const std::vector<Sample>& data) {
    if (data.size() == 0) {
        return 0;
    }

    int class0 = 0;
    int class1 = 0;
    int class2 = 0;

    for (int i = 0; i < data.size(); i++) {
        if (data[i].label == 0) {
            class0++;
        } else if (data[i].label == 1) {
            class1++;
        } else if (data[i].label == 2) {
            class2++;
        }
    }

    if (class0 >= class1 && class0 >= class2) {
        return 0;
    } else if (class1 >= class0 && class1 >= class2) {
        return 1;
    } else {
        return 2;
    }
}


// selecting the random subset of features, to be used in the random forest (extension)
std::vector<int> getRandomFeatures(int numFeatures) {
    std::vector<int> features;

    while (features.size() < numFeatures) {
        int f = rand() % 10;

        bool exists = false;
        for (int i = 0; i < features.size(); i++) {
            if (features[i] == f) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            features.push_back(f);
        }
    }

    return features;
}

//returning all of the ten feature indexes
std::vector<int> getAllFeatures() {
    std::vector<int> features;
    for (int i = 0; i < 10; i++) {
        features.push_back(i);
    }
    return features;
}

Node* buildOneSplitTree(const std::vector<Sample>& trainingData) {
    int bestFeature = findBestFeature(trainingData, getAllFeatures());

    std::vector<Sample> leftData;
    std::vector<Sample> rightData;
    splitDataByFeature(trainingData, bestFeature, leftData, rightData);

    Node* root = new Node();
    root->data = trainingData;
    root->isLeaf = false;
    root->splitFeature = bestFeature;

    Node* leftLeaf = new Node();
    leftLeaf->isLeaf = true;
    leftLeaf->predictedClass = majorityClass(leftData);

    Node* rightLeaf = new Node();
    rightLeaf->data = rightData;
    rightLeaf->isLeaf = true;
    rightLeaf->predictedClass = majorityClass(rightData);

    root->left = leftLeaf;
    root->right = rightLeaf;

    return root;
}

// following the decision tree and predicting the class of one sample 
int predict(Node* node, const Sample& sample) {
    if (node->isLeaf) {
        return node->predictedClass;
    }

    if (sample.features[node->splitFeature] == 0) {
        return predict(node->left, sample);
    } else {
        return predict(node->right, sample);
    }
}

// calculating the accuracy of a decision tree on the test data
double testAccuracy(Node* root, const std::vector<Sample>& testData) {
    int correct = 0;

    for (int i = 0; i < testData.size(); i++) {
        int predicted = predict(root, testData[i]);
        if (predicted == testData[i].label) {
            correct++;
        }
    }

    return static_cast<double>(correct) / testData.size();
}

// creating 2 new child leaves by converting 1 leaf into a rule node
bool splitLeaf(Node* node, int bestFeature) {
    std::vector<Sample> leftData;
    std::vector<Sample> rightData;

    splitDataByFeature(node->data, bestFeature, leftData, rightData);

    if (bestFeature == -1 || leftData.size() == 0 || rightData.size() == 0) {
        return false;
    }

    node->isLeaf = false;
    node->splitFeature = bestFeature;

    Node* leftLeaf = new Node();
    leftLeaf->data = leftData;
    leftLeaf->predictedClass = majorityClass(leftData);

    Node* rightLeaf = new Node();
    rightLeaf->data = rightData;
    rightLeaf->predictedClass = majorityClass(rightData);

    node->left = leftLeaf;
    node->right = rightLeaf;

    return true;
}

// splitting the best leaf from the priority queue repeatedly to build the decision tree
void buildTree(Node* root, int maxSplits, const std::vector<int>& allowedFeatures) {
    std::priority_queue<QueueItem> pq;

    SplitCandidate rootSplit = findBestSplit(root->data, allowedFeatures);
    if (rootSplit.feature != -1) {
        pq.push({rootSplit.score, rootSplit.feature, root});
    }

    int splits = 0;

    while (!pq.empty() && splits < maxSplits) {
        QueueItem item = pq.top();
        pq.pop();

        Node* node = item.node;

        if (!node->isLeaf) continue;
        if (item.score >= 0.0) continue;

        if (!splitLeaf(node, item.feature)) continue;
        splits++;

        if (node->left->data.size() > 0) {
            SplitCandidate leftSplit = findBestSplit(node->left->data, allowedFeatures);
            if (leftSplit.feature != -1) {
                pq.push({leftSplit.score, leftSplit.feature, node->left});
            }
        }

        if (node->right->data.size() > 0) {
            SplitCandidate rightSplit = findBestSplit(node->right->data, allowedFeatures);
            if (rightSplit.feature != -1) {
                pq.push({rightSplit.score, rightSplit.feature, node->right});
            }
        }
    }
}

// using all trees to vote and selecting the most common predicted class
int predictForest(const std::vector<Node*>& forest, const Sample& sample) {
    int votes[3] = {0, 0, 0};

    for (int i = 0; i < forest.size(); i++) {
        int pred = predict(forest[i], sample);
        votes[pred]++;
    }

    int bestClass = 0;
    if (votes[1] > votes[bestClass]) bestClass = 1;
    if (votes[2] > votes[bestClass]) bestClass = 2;

    return bestClass;
}

// calculating the accuracy of the full random forest on the test data
double testForestAccuracy(const std::vector<Node*>& forest, const std::vector<Sample>& testData) {
    int correct = 0;

    for (int i = 0; i < testData.size(); i++) {
        int predicted = predictForest(forest, testData[i]);
        if (predicted == testData[i].label) {
            correct++;
        }
    }

    return (double)correct / testData.size();
}

int main() {
    srand(1);

    std::ifstream trainFile("training.dat");
    std::ifstream testFile("testing.dat");

    if (!trainFile.is_open()) {
        std::cout << "Could not open training.dat" << std::endl;
        return 1;
    }

    if (!testFile.is_open()) {
        std::cout << "Could not open testing.dat" << std::endl;
        return 1;
    }

    std::vector<Sample> trainingData;
    std::vector<Sample> testingData;

    Sample s;

    // reading the training data
    while (
        trainFile >> s.features[0] >> s.features[1] >> s.features[2] >> s.features[3] >> s.features[4]
                  >> s.features[5] >> s.features[6] >> s.features[7] >> s.features[8] >> s.features[9]
                  >> s.label
    ) {
        trainingData.push_back(s);
    }
    
    // reading the testing data
    while (
        testFile >> s.features[0] >> s.features[1] >> s.features[2] >> s.features[3] >> s.features[4]
                 >> s.features[5] >> s.features[6] >> s.features[7] >> s.features[8] >> s.features[9]
                 >> s.label
    ) {
        testingData.push_back(s);
    }

    std::cout << "Training rows stored: " << trainingData.size() << std::endl;
    std::cout << "Testing rows stored: " << testingData.size() << std::endl;

int singleTreeSplits = 100;
int numTrees = 50;
int forestSplits = 100;
int featuresPerTree = 4;

Node* singleTree = new Node();
singleTree->data = trainingData;
singleTree->predictedClass = majorityClass(trainingData);
buildTree(singleTree, singleTreeSplits, getAllFeatures());

double singleTreeAccuracy = testAccuracy(singleTree, testingData);
std::cout << "Single Decision Tree Accuracy: " << singleTreeAccuracy << std::endl;

std::vector<Node*> forest;
double totalTreeAccuracy = 0.0;
double lowestTreeAccuracy = 1.0;
double highestTreeAccuracy = 0.0;

// creating multiple trees and storing them into one forest
for (int t = 0; t < numTrees; t++) {

    std::vector<int> features = getRandomFeatures(featuresPerTree);
    Node* root = new Node();
    root->data = trainingData;
    root->predictedClass = majorityClass(trainingData);
    buildTree(root, forestSplits, features);
    forest.push_back(root);

    double accuracy = testAccuracy(root, testingData);
    totalTreeAccuracy += accuracy;

    if (accuracy < lowestTreeAccuracy) {
        lowestTreeAccuracy = accuracy;
    }

    if (accuracy > highestTreeAccuracy) {
        highestTreeAccuracy = accuracy;
    }

    std::cout << "Tree " << t+1
              << " | Accuracy: " << accuracy << std::endl;
}

double averageTreeAccuracy = totalTreeAccuracy / numTrees;
std::cout << "Average Individual Tree Accuracy: " << averageTreeAccuracy << std::endl;
std::cout << "Lowest Individual Tree Accuracy: " << lowestTreeAccuracy << std::endl;
std::cout << "Highest Individual Tree Accuracy: " << highestTreeAccuracy << std::endl;

// testing the final combined forest using majority voting
double forestAccuracy = testForestAccuracy(forest, testingData);
std::cout << "Forest Accuracy: " << forestAccuracy << std::endl;
    return 0;
}
