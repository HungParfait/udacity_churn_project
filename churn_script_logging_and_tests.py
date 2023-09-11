import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='a',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
    '''
    test perform_eda
    '''
    try:
        df_test = cls.import_data('./data/bank_data.csv')
        perform_eda(df_test)
        assert os.path.exists('./images/eda/Churn.png')
        assert os.path.exists('./images/eda/Customer_Age.png')
        assert os.path.exists('./images/eda/Marital_Status.png')
        assert os.path.exists('./images/eda/Total_Trans_Ct.png')
        assert os.path.exists('./images/eda/heatmap.png')
        logging.info("Testing perform_eda: DONE - SUCCESS")

    except AssertionError as err:
        logging.error(
            "FAILED: Testing perform_eda")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df_test = cls.import_data('./data/bank_data.csv')
    cls.perform_eda(df_test)

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(df_test)
        assert x_train.shape[0] and y_train.shape[0] > 0 \
            and x_test.shape[0] and y_test.shape[0] > 0
        logging.info("SUCCESS: Testing perform_feature_engineering")

    except AssertionError as err:
        logging.error("FAILED: Testing perform_feature_engineering!")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df_test = cls.import_data('./data/bank_data.csv')
    cls.perform_eda(df_test)

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        df=df_test)

    try:
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.exists('models/logistic_model.pkl')
        assert os.path.exists('models/rfc_model.pkl')
        logging.info("SUCCESS: Testing train_models")

    except AssertionError as err:
        logging.error("FAILED: Testing train_models!")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)




