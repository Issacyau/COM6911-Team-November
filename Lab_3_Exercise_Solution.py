from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


spark = SparkSession.builder \ ### entry point of spark
        .master("local[2]") \
        .appName("Lab 3 Exercise") \
        .config("spark.local.dir","/fastdata/act20ly") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")



from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# load in ratings data
ratings = spark.read.load('../Data/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
ratings.show(20,False)
myseed=6012

# split
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()

# define model
als = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
# define evaluator
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")

# run model with default rank (which is 10)
model = als.fit(training)


def run_model(_train, _test, _als, _evaluator):
    
    model = _als.fit(_train)
    predictions = model.transform(_test)
    rmse = _evaluator.evaluate(predictions)
    print(f"rank {_als.getRank()} Root-mean-square error = {rmse}")
    return rmse

# run model for 5 times    
ranks = [5,10,15,20,25]
results = []
for _rank in ranks:
    als.setRank(_rank)
    results.append(run_model(training, test, als, evaluator))

# plot results
fig, ax = plt.subplots()

rects = ax.bar([str(r) for r in ranks], results, label = "rmse")

ax.set_ylabel('RMSE result')
ax.set_title("RMSE by ranks")
ax.yaxis.set_data_interval(min(results), max(results),True)
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.savefig("../Output/Lab3_plot.png")


# select a user
### distinct() -> return dataset with unique element of the source dataset
users = ratings.select(als.getUserCol()).distinct().sample(withReplacement =  False, fraction = 0.1, seed = myseed).limit(1)
users.show()
# get recomendations from model
userSubsetRecs = model.recommendForUserSubset(users, 5)
userSubsetRecs.show(1, False)
# get movie_id
movies = userSubsetRecs.collect()[0].recommendations
movies = [row.movieId for row in movies]
print(movies)

# loading movies.csv
movie_data = spark.read.load('../Data/ml-latest-small/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
movie_data.show(20, False)

# find movie according to movie_id
for movie_id in movies:
    _data = movie_data.filter(movie_data.movieId == f"{movie_id}").collect()
    _data = _data[0]    
    print(f"movie id: {movie_id} \t title: {_data.title} \t genres: {_data.genres}")
