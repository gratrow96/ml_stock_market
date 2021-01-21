from gradient_boosting_classifier import gradient_boosting_tree
from naive_bayes import naive_bayes_classifier

class market_analysis_controller:
    """docstring for market_analysis_controller"""
    def __init__(self) -> None:
        pass

def main():
    """
    """
    print('\nGradient Boosting Classification:')
    # so far best learning rate is .001
    gradient_b_classifier = gradient_boosting_tree('exponential', .001, 1000, 100,
                                            99, 98, 'friedman_mse', .1)
    gradient_b_classifier.split_data_classification('NVDA_long.csv')
    gradient_b_classifier.gradient_boost_classify()
    # gradient_b_classifier.many_classifications(10)

    print('\nNaive Bayes Classification:')
    # gaussian = naive_bayes_classifier(.001, 'gaussian')
    # gaussian.split_data_classification('NVDA_long.csv')
    # gaussian.run_naive_bayes()
    # # gaussian.many_naive_bayes(1000)

    # multinomial = naive_bayes_classifier(.001, 'multinomial')
    # multinomial.split_data_classification('NVDA_long.csv')
    # multinomial.run_naive_bayes()
    # # multinomial.many_naive_bayes(1000)

    # complement = naive_bayes_classifier(.001, 'complement')
    # complement.split_data_classification('NVDA_long.csv')
    # complement.run_naive_bayes()
    # # complement.many_naive_bayes(1000)

    # bernoulli = naive_bayes_classifier(.001, 'bernoulli')
    # bernoulli.split_data_classification('NVDA_long.csv')
    # bernoulli.run_naive_bayes()
    # # bernoulli.many_naive_bayes(1000)

if __name__ == '__main__':
	main()