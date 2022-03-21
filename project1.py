#--------------------------------------------------------------------------------------------------------------------#
#                                                                                                                    #
#                    Contains all necessary functions to implement and complete the first project                    #
#                                                                                                                    #
#--------------------------------------------------------------------------------------------------------------------#


# Helper Functions (given)

def get_order(n_samples):
    """
    Opens txt files and gives order to samples
    
    Args:
        n_samples - A numpy matrix describing the given data. Each row represents a single data point.
    
    Returns: An ordered list of the samples
    """
    
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    
    return (preds == targets).mean()


      
# Loss Functions

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification parameters theta and theta_0.
    
    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the datapoint.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    
    Returns: A real number representing the hinge loss associated with the given data point and parameters.
    """
    
    y = np.dot(theta, feature_vector) + theta_0
    return max(0, 1 - y * label)
 

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification parameters theta and theta_0.
    
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row represents a single data point.
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    
    Returns: A real number representing the hinge loss associated with the given dataset and parameters.
    This number should be the average hinge loss across all of the points in the feature matrix.
    """
    
    ys = np.dot(theta, feature_matrix.T) + theta_0
    loss = 1 - ys * labels
    loss[loss < 0] = 0  # Returns zero if loss is negative 
                        # Alternative: check length of inputs and return similar size matrix of zeros:
                        # loss = np.maximum(1 - ys * labels, np.zeros(len(labels)))

    return np.mean(loss)


  
# Classifier Algorithms

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Properly updates the classification parameters, theta and theta_0, on a single step of the perceptron algorithm.
    
    Because of numerical instabilities, it is preferable to identiy zero with a small range of values close to zero [-epsilon, epsilon].
    This means that when x is a float, "x=0" should be checked with |x| < epsilon
    
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron algorithm before this update.
    
    Returns: A tuple where the first element is a numpy array with the value of theta after the current update has completed
    and the second element is a real valued number with the value of theta_0 after the current updated has completed.
    """

    # Tolerance for numerical instabilities / error margin for rounding errors
    epsilon = 1e-7

    x = float(label * (np.dot(current_theta, feature_vector) + current_theta_0))

    if abs(x) < epsilon or x < 0:   # Checking whether to update with the epsilon condition
        current_theta = current_theta + label * feature_vector
        current_theta_0 = current_theta_0 + label

    return (current_theta, current_theta_0)

  
  def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data with T iterations.
    
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row represents a single data point.
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm should iterate through the feature matrix.
    
    Returns: A tuple where the first element is a numpy array with the value of theta,
    the linear classification parameter, after T iterations through the feature matrix
    and the second element is a real number with the value of theta_0, the offset classification parameter,
    after T iterations through the feature matrix.
    """
    
    #Initializing parameters to zero according to the dimensions of the inputs
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0.0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]): # Ordering the rows of the feature matrix
            current_theta, current_theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], current_theta, current_theta_0)

    return (current_theta, current_theta_0)
  
  
  def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data with T iterations.
    Adds a modification to the original perceptron algorithm by returning an average of the parameters of the n steps as it updates.
    
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row represents a single data point.
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm should iterate through the feature matrix.
    
    Returns: A tuple where the first element is a numpy array with the value of the average theta, the linear classification parameter,
    found after T iterations through the feature matrix and the second element is a real number with the value of the average theta_0,
    the offset classification parameter, found after T iterations through the feature matrix.
    """
    
    # Initializing parameters to zero according to the dimensions of the inputs
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0.0

    # Keeping track of the sum through the loops
    theta_sum = np.zeros(feature_matrix.shape[1])
    theta_0_sum = 0.0

    n = feature_matrix.shape[0]     # Number of samples (rows)

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            current_theta, current_theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], current_theta, current_theta_0)

            theta_sum = theta_sum + current_theta
            theta_0_sum = theta_0_sum + current_theta_0

    theta_avg = (1/(n*T)) * theta_sum
    theta_0_avg = (1/(n*T)) * theta_0_sum

    return (theta_avg, theta_0_avg)
  
  
  def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a single step of the Pegasos algorithm.
    
    The Pegasos algorithm mixes together regularization, hinge loss, subgradient updates, and a decaying learning rate.
    Paper: Shai Shalev-Shwartz et al., (2000), "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM".
    
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos algorithm before this update.
        current_theta_0 - The current theta_0 being used by the Pegasos algorithm before this update.
    
    Returns: A tuple where the first element is a numpy array with the value of theta after the current update has completed
    and the second element is a real valued number with the value of theta_0 after the current updated has completed.
    """
    
    x = float(label * (np.dot(current_theta, feature_vector) + current_theta_0))

    if x <= 1.0: 
        current_theta = (1-eta * L) * current_theta + eta * label * feature_vector # Eta decreases over time and lambda is a regularization parameter
        current_theta_0 = current_theta_0 + eta * label
    else:
        current_theta = (1-eta * L) * current_theta

    return (current_theta, current_theta_0)
  
  
  def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data with T iterations.
    For each update, set learning rate = 1/sqrt(t), where t is a counter for the number of updates performed so far
    (between 1 and nT inclusive).
    
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row represents a single data point.
        labels - A numpy array where the kth element of the array is the correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos algorithm parameters.
    
    Returns: A tuple where the first element is a numpy array with the value of the theta, the linear classification parameter,
    found after T iterations through the feature matrix and the second element is a real number with the value of the theta_0,
    the offset classification parameter, found after T iterations through the feature matrix.
    """

    # Initializing parameters to zero according to the dimensions of the inputs
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0.0
    
    # Adding counter
    c = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):  # Getting the order of samples with the helper function
            eta = 1 / np.sqrt(c)  # Update learning rate eta 
            c += 1  # Add one to counter each time

            # Update parameters theta and theta_0
            current_theta, current_theta_0 = pegasos_single_step_update(feature_matrix[i, :], labels[i], L, eta, current_theta, current_theta_0)

    return (current_theta, current_theta_0)

  
# Functions for Classification and Evaluation of Accuracy

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of data points.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row represents a single data point.
        theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    
    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted classification of the kth row of the feature matrix
    using the given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Tolerance for numerical instabilities / error margin for rounding errors
    epsilon = 1e-7

    predictions = np.dot(theta, feature_matrix.T) + theta_0
    predictions[predictions > 0.0] = 1  # Only greater than zero is considered a positive classification
    predictions[predictions < 0.0] = -1
    predictions[abs(predictions) < epsilon] = -1

    return predictions

  
def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels,**kwargs):
    """
    Trains a linear classifier and computes accuracy.
    
    Args:
        classifier - A classifier function that takes arguments (feature matrix, labels, **kwargs)
        train_feature_matrix - A numpy matrix describing the training data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array is the correct classification of the kth row of the training feature matrix.
        val_labels - A numpy array where the kth element of the array is the correct classification of the kth row of the validation feature matrix.
        **kwargs - Additional named arguments to pass to the classifier (e.g. T or L).
    
    Returns: A tuple in which the first element is the (scalar) accuracy of the trained classifier on the training data
    and the second element is the accuracy of the trained classifier on the validation data.
    """
    # Training the classifier to get parameters theta and theta_0
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)

    # Using the parameters to get predictions for training and validation sets
    pred_train = classify(train_feature_matrix, theta, theta_0)
    pred_val = classify(val_feature_matrix, theta, theta_0)

    # Comparing predictions with labels to get accuracy
    train_accuracy = accuracy(pred_train, train_labels)
    val_accuracy = accuracy(pred_val, val_labels)

    return (train_accuracy, val_accuracy)

  
# Functions for NLP

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    
    Problem 9: Feature Engineering  - remove stopwords from dictionary
    Try to implement stop words removal in your feature engineering code. Specifically,
    load the file stopwords.txt, remove the words in the file from your dictionary,
    and use features constructed from the new dictionary to train your model 
    and make predictions.
    """
    
    # Reading stopwords from file and saving them
    with open("data\stopwords.txt", 'r', encoding='utf8') as stop_text:
        stop_words = stop_text.read()
        stop_words = stop_words.replace("\n", " ").split()

    dictionary = {}  # maps each word to a unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stop_words: # Check if word also not in stopwords
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    
    Problem 9: Feature Engineering - Change binary features to counts features.
    When you compute the feature vector of a word, use its count in each document
    rather than a binary indicator.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1 # Adding a + to change from binary to counts
    return feature_matrix
