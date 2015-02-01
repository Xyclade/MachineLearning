#Machine Learning for developers 


Most developers these days have heard of machine learning, but when trying to find an 'easy' way into this technique, most people find themselves getting scared off by the abstractness of the concept of *Machine Learning* and terms as   [*regression*](http://en.wikipedia.org/wiki/Regression_analysis), [*unsupervised learning*](http://en.wikipedia.org/wiki/Unsupervised_learning), [*Probability Density Function*](http://en.wikipedia.org/wiki/Probability_density_function) and many other definitions. 

And then there are books such as [*Machine Learning for Hackers*](http://shop.oreilly.com/product/0636920018483.do) and [*An Introduction to Statistical Learning with Applications in R*](http://www-bcf.usc.edu/~gareth/ISL/)  who use programming language [R](http://www.r-project.org) for their examples. 

However R is not really a programming language in which one writes programs for everyday use such as is done with for example Java, C#, Scala etc. This is why in this blog, Machine learning will be introduced using libraries for java/scala and C# which are languages that almost every developer has worked with once during their study or carreer.

Note that in this blog, 'new' definitions are hyperlinked such that if you want, you **can** read more regarding that specific topic, but you are not obliged to do this in order to be able to work through the examples. However the section ['The global idea of machine learning'](#The global idea of machine learning) helps making things a lot more clear when working through the examples and is advised to be read on beforehand in case you are completely new to Machine Learning.




##Practical Examples

The examples will start off with the most simple and intuitive [*Classification*](http://en.wikipedia.org/wiki/Statistical_classification) Machine learning Algorithm there is: [*K-Nearest Neighbours*](http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

###Labeling ISPs based on their Down/Upload speed (K-NN using Smile in Scala)

The goal of this section is to use the K-NN implementation from [Smile](https://github.com/haifengl/smile) in Scala to classify download/upload speed pairs as [ISP](http://en.wikipedia.org/wiki/Internet_service_provider) Alpha (represented by 0) or Beta (represented by 1).  

To start with this example I assume you created a new Scala project in your favorite IDE, and downloaded and added the [Smile Machine learning](https://github.com/haifengl/smile/releases)  and its dependency [SwingX](https://java.net/downloads/swingx/releases/) to this project. As final assumption you also downloaded the [example data](https://github.com/Xyclade/MachineLearning/raw/Master/Example%20Data/KNN_Example_1.csv).

The first step is to load the CSV data file. As this is no rocket science, I provide the code for this without further explanation:

```scala

object KNNExample {
    def main(args: Array[String]): Unit = {
    val basePath = "/.../KNN_Example_1.csv"
    val testData = GetDataFromCSV(new File(basePath))    
    }
    
  def GetDataFromCSV(file: File): (Array[Array[Double]], Array[Int]) = {
    val source = scala.io.Source.fromFile(file)
    val data = source.getLines().drop(1).map(x => GetDataFromString(x)).toArray
    source.close()
    val dataPoints = data.map(x => x._1)
    val classifierArray = data.map(x => x._2)
    return (dataPoints, classifierArray)        
  }
  
  def GetDataFromString(dataString: String): (Array[Double], Int) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')

    //Extract the values from the strings
    val xCoordinate: Double = dataArray(0).toDouble
    val yCoordinate: Double = dataArray(1).toDouble
    val classifier: Int = dataArray(2).toInt

    //And return the result in a format that can later easily be used to feed to Smile
    return (Array(xCoordinate, yCoordinate), classifier)
  }
}
```

First thing you might wonder now is *why the frick is the data formatted that way*. Well, the separation between dataPoints and their label values is for easy splitting between testing and training data, and because the API expects the data this way for both executing the KNN algorithm and plotting the data. Secondly the datapoints stored as an ```scala Array[Array[Double]]``` is done to support datapoints in more than just 2 dimensions.

Given the data the first thing to do next is to see what the data looks like. For this Smile provides a nice Plotting library. In order to use this however, the application should be changed to a Swing application. Additionally the data should be fed to the plotting library to get a JPane with the actual plot. After changing your code it should like this:

 ```scala
 
 object KNNExample extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "KNN Example!"
    val basePath = "/Users/mikedewaard/ML_for_Hackers/10-Recommendations/data/example_data.csv"


    val test_data = GetDataFromCSV(new File(basePath))

    val plot = ScatterPlot.plot(test_data._1, test_data._2, '@', Array(Color.red, Color.blue));
    peer.setContentPane(plot)
    size = new Dimension(400, 400)

  }
  ...
  ```
 
The idea behind plotting the data is to verify whether KNN is a fitting Machine Learning algorithm for this specific set of data. In this case the data looks as follows:

![KNN Data plot](https://github.com/Xyclade/MachineLearning/raw/KNN_Example/Images/KNNPlot.png =200x200)

In this plot you can see  that the blue and red points seem to be mixed in the area (3 < x < 5) and (5 < y < 7.5). If the groups were not mixed at all, but were two separate clusters, a [regression](#regression) algorithm  such as in the example [Page view prediction with regression](#Page view prediction with regression) could be used instead. However, since the groups are mixed the KNN algorithm is a good choice as fitting a linear decision boundary would cause a lot of false classifications in the mixed area.

Given this choice to use KNN to be a good one, lets continue with the actual Machine Learning part. For this the GUI is ditched since it does not really add any value. Recall from the section [*The global idea of Machine Learning*](#The global idea of machine learning) that in machine learning there are 2 key parts: Prediction and Validation. First we will look at the Validation, as using a model without any validation is never a good idea. The main reason to validate the model here is to prevent [overfitting](#overfitting). However, before we even can do validation, a *correct* K should be chosen. 

The drawback is that there is no golden rule for finding the correct K. However finding a K that allows for most datapoints to be classified correctly can be done by looking at the data. Additionally The K should be picked carefully to prevent undecidability by the algorithm. Say for example ```K=2```, and the problem has 2 labels, then when a point is between both labels, which one should the algorithm pick. There is a *rule of Thumb* that K should be the square root of the number of features (on other words the number of dimensions). In our example this would be ```K=1```, but this is not really a good idea since this would lead to higher false-classifications around decision boundaries. Picking ```K=2``` would result in the error regarding our two labels, thus picking ```K=3``` seems like a good fit for now.

For this example we do [2-fold Cross Validation](http://en.wikipedia.org/wiki/Cross-validation_(statistics) ). In general 2-fold Cross validation is a rather weak method of model Validation, as it splits the dataset in half and only validates twice, which still allows for overfitting, but since the dataset is only 100 points, 10-fold (which is a stronger version) does not make sense, since then there would only be 10 datapoints used for testing, which would give a skewed error rate.

```scala

  def main(args: Array[String]): Unit = {
    val basePath = "/Users/mikedewaard/ML_for_Hackers/10-Recommendations/data/example_data.csv"
    val testData = GetDataFromCSV(new File(basePath))

    //Define the amount of rounds, in our case 2 and initialize the cross validation
    val validationRounds = 2;
    val cv = new CrossValidation(testData._2.length, validationRounds);
    //Then for each round
    for (i <- 0 to validationRounds - 1) {

      //Generate a subset of data points and their classifiers for Training
      val dpForTraining = testData._1.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1);
      val classifiersForTraining = testData._2.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1);

      //And the corresponding subset of datapoints and their classifiers for testing
      val dpForTesting = testData._1.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1);
      val classifiersForTesting = testData._2.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1);

      //Then generate a Model with KNN with K = 3 
      val knn = KNN.learn(dpForTraining, classifiersForTraining, 3);

      //And for each test data point make a prediction with the model
      val predictions = dpForTesting.map(x => knn.predict(x))

      //Finally evaluate the predictions as correct or incorrect and count the amount of wrongly classified data points.
      var error = 0.0;
      for (j <- 0 to predictions.length - 1) {
        if (predictions(j) != classifiersForTesting(j)) {
          error += 1
        }
      }
      println("false prediction rate: " +  error / predictions.length * 100 + "%")
    }
  }
  ```
If you execute this code serveral times you might notice the false prediction rate to fluctuate quite a bit. This is due to the random samples taken for training and testing. When this random sample is taken a bit unforunate, the error rate becomes much higher while when taking a good random sample, the error rate could be extremely low. 

Unfortunately I cannot provide you with a golden rule to when your model was trained with the best possible training set. One would say the model with the least error rate is always the best, but when you recall the term [overfitting](#overfitting) picking this particular model might perform really bad on future data. This is why having a large enough and representative dataset is key to a good Machine Learning application. However when aware of this issue, you could implement manners to keep updating your model based on new data and known correct classifications.


Let's suppose you implement the KNN into your application, then you should have gone through the following steps. First you took care of getting the training and testing data. Next up you generated and validated serveral models and picked the model which gave the best results. After these steps you can finally do predictions using your ML implementations:


```scala
 
 	  val result = knn.predict(unknownDatapoint);

      if (result == 0)
      {
        println("Internet Service Provider Alpha")
      }
      else if (result == 1)
      {
        println("Internet Service Provider Beta")
      }
      else
      {
        println("Unexpected prediction")
      }
    }

```

These predictions can then be used to present to the users of your system, for example as friend suggestion on a social networking site. The feedback the user gives on these predictions is valuable and should thus be fed into the system for updating your model.
 


###Classifying Email as Spam or Ham (Naive Bayes)
The goal of this section is to use the Naive Bayes implementation from [Smile](https://github.com/haifengl/smile) in Scala to classify emails as Spam or Ham based on their content.  

To start with this example I assume you created a new Scala project in your favorite IDE, and downloaded and added the [Smile Machine learning](https://github.com/haifengl/smile/releases)  and its dependency [SwingX](https://java.net/downloads/swingx/releases/) to this project. As final assumption you also downloaded and extracted the [example data](https://github.com/Xyclade/MachineLearning/raw/Master/Example%20Data/NaiveBayes_Example_1.zip). This example data comes from the [SpamAssasins public corpus](http://spamassasin.apache.org/publiccorpus/). 

As with every machine learning implementation, the first step is to load in the training data. However in this example we are taking it 1 step further into machine learning. In the KNN examples we had the download and upload speed as [features](#features), which was rather easy as they where numbers and the only features available. For spam classification it is not completely trivial what to use as features. 

In this example we will use the content of the email as feature. By this we mean, we will select  the features (words in this case) from the content of the training set of emails. In order to be able to do this, we need to build a [Term Document Matrix (TDM)](http://en.wikipedia.org/wiki/Document-term_matrix). We could use a library for it, but in order to gain more insight in why as to use it, let's build it ourselves, as this also gives us all freedom in properly sellecting the features:

```
scala

//Insert code snippet for loading the filenames + the messages

```
With the data loaded building the TDM can begin:

```
scala
//TDM class implementation

```

```
scala
//Showing how the TDM works and how to change round the outputs., including the top spammy and ham words.

```

Ok now that we have some insight in what are 'spammy' words and what are typical 'ham-words', we can decide on building a feature-set which we can then use in the Naive Bayes algorithm for creating the classifier. Note: In most cases it is always better to include **more** features, however performance becomes an issue when having tons of features. This is why in the field, developers tend to drop features that do not have a significant impact, purely for performance reasons.

For now we will select the top **xx** spammy words based on occurence(thus not  frequency) and do the same for ham words and combine this into 1 set of words which we can feed into the bayes algorithm.


```
scala
//Add the code for getting the tdm data and combining it into a feature bag.

```

Given this feature bag, and a set of test data, we can start training the algorithm. For this we can chose a few different models: 'general',  'multinominal' and bernoulli. In this example we focus on the multinominal but feel free to try out the other model types as well.


```
scala
//Add the code for creating the bayes classifier, including the training part

```

Now that we have the trained model, we can once again do some validation. However, in the example data we already made a separation between easy and hard ham, and spam, thus we will not apply the cross validation, but rather validate the model on hard-ham.

```
scala
//Add code for model validation

```

###Page view prediction with regression




##The global idea of machine learning
The term 'Machine learning' is known by almost everyone, however almost no-one I spoke could really explain what it is in **_one_** or even serveral sentences.

*Explanation of machine learning, preferably with an graphical image*


In the upcoming subsections the most important notions you need to be aware off when practicing machine learning are (briefly) explained.

###Features
A feature (in the field of machine learning) is a property on which a [Model](#Model) is trained. Say for example that you classify emails as spam/ham based on the frequency of the word 'Buy' and 'Money'. Then these words are features. If you would use machine learning to predict whether one is a friend of you, the amount of 'common' friends could be a feature.

###Model
When one talks about machine learning, often the term *model* is mentioned. The model is the result of any machine learning method and the algorithm used within this method. This model can be used to make predictions in [supervised](#Supervised Learning), or to retrieve clusterings in [unsupervised learning](#Unsupervised learning).

###Learning methods
In the field of machine learning there are two leading ways of learning, namely [Supervised learning](http://en.wikipedia.org/wiki/Supervised_learning) and  [Unsupervised learning](http://en.wikipedia.org/wiki/Unsupervised_learning). A brief introduction is neccesary when you want to use Machine learning in your applications, as picking the right machine learning approach is an important but sometimes also a little tedious process.

####Supervised Learning
The principle of supervised learning can be used to solve many problem types. In this blog however we will stick to [Classification](#Classification) and [Regression](#Regression) as this covers most of the problems one wants to solve in their every day application.


#####Classification
The problem of classification within the domain of Supervised learning is relatively simple. Given a set of labels, and some data that already received the correct labels, we want to be able to *predict* labels for new data that we did not label yet. However, before thinking of your data as a classification problem, you should look at what the data looks like. If there is a clear structure in the data such that you can easily draw a regression line it might be better to use a [regression](#Regression) algorithm instead. Given the data does not fit to a regression line, or when performance becomes an issue, classification is a good alternative.

An example of a classification problem would be to classify emails as Ham or Spam based on their content. Given a training set in which emails are labeled Ham/Spam, a classification algorithm can be used to train a [Model](#Model). This model can then be used to predict for future emails whether they are Ham or Spam. A typical example of a classification algorithm is the [K-NN algorithm](#Labeling ISPs based on their Down/Upload speed (K-NN using Smile in Scala)). Another more commonly used example of a classification problem is [Classifying Email as Spam or Ham](###Classifying Email as Spam or Ham (Naive Bayes)) which is also one of the examples written on this blog.

#####Regression


####Unsupervised Learning



###Validation techniques

#### Overfitting
#### Underfitting

####Precision

####Recall

###Exploratory Data Analysis








