#Machine Learning for developers 


Most developers these days have heard of machine learning, but when trying to find an 'easy' way into this technique, most people find themselves getting scared off by the abstractness of the concept of *Machine Learning* and terms as   [*regression*](http://en.wikipedia.org/wiki/Regression_analysis), [*unsupervised learning*](http://en.wikipedia.org/wiki/Unsupervised_learning), [*Probability Density Function*](http://en.wikipedia.org/wiki/Probability_density_function) and many other definitions. 

And then there are books such as [*Machine Learning for Hackers*](http://shop.oreilly.com/product/0636920018483.do) and [*An Introduction to Statistical Learning with Applications in R*](http://www-bcf.usc.edu/~gareth/ISL/)  who use programming language [R](http://www.r-project.org) for their examples. 

However R is not really a programming language in which one writes programs for everyday use such as is done with for example Java, C#, Scala etc. This is why in this blog, Machine learning will be introduced using [Smile](https://github.com/haifengl/smile), a machine learning library that can be used both in Java and Scala,which are languages that almost every developer has worked with once during their study or career.

Note that in this blog, 'new' definitions are hyperlinked such that if you want, you **can** read more regarding that specific topic, but you are not obliged to do this in order to be able to work through the examples. However the section ['The global idea of machine learning'](#The-global-idea-of-machine-learning) helps making things a lot more clear when working through the examples and is advised to be read on beforehand in case you are completely new to Machine Learning.



##Practical Examples

The examples will start off with the most simple and intuitive [*Classification*](http://en.wikipedia.org/wiki/Statistical_classification) Machine learning Algorithm there is: [*K-Nearest Neighbours*](http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

###Labeling ISPs based on their Down/Upload speed (KNN using Smile in Scala)

The goal of this section is to use the K-NN implementation from  in Scala to classify download/upload speed pairs as [ISP](http://en.wikipedia.org/wiki/Internet_service_provider) Alpha (represented by 0) or Beta (represented by 1).  

To start with this example I assume you created a new Scala project in your favourite IDE, and downloaded and added the [Smile Machine learning](https://github.com/haifengl/smile/releases)  and its dependency [SwingX](https://java.net/downloads/swingx/releases/) to this project. As final assumption you also downloaded the [example data](https://github.com/Xyclade/MachineLearning/raw/Master/Example%20Data/KNN_Example_1.csv).

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

First thing you might wonder now is *why the frick is the data formatted that way*. Well, the separation between dataPoints and their label values is for easy splitting between testing and training data, and because the API expects the data this way for both executing the KNN algorithm and plotting the data. Secondly the datapoints stored as an ```Array[Array[Double]]``` is done to support datapoints in more than just 2 dimensions.

Given the data the first thing to do next is to see what the data looks like. For this Smile provides a nice Plotting library. In order to use this however, the application should be changed to a Swing application. Additionally the data should be fed to the plotting library to get a JPane with the actual plot. After changing your code it should like this:

 ```scala
 
 object KNNExample extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "KNN Example"
    val basePath = "/.../KNN_Example_1.csv"

    val test_data = GetDataFromCSV(new File(basePath))

    val plot = ScatterPlot.plot(test_data._1, test_data._2, '@', Array(Color.red, Color.blue))
    peer.setContentPane(plot)
    size = new Dimension(400, 400)

  }
  ...
  ```
 
The idea behind plotting the data is to verify whether KNN is a fitting Machine Learning algorithm for this specific set of data. In this case the data looks as follows:

<img src="./Images/KNNPlot.png" width="300px" height="300px" />

In this plot you can see  that the blue and red points seem to be mixed in the area (3 < x < 5) and (5 < y < 7.5). If the groups were not mixed at all, but were two separate clusters, a [regression](#regression) algorithm  such as in the example [Page view prediction with regression](#Page-view-prediction-with-regression) could be used instead. However, since the groups are mixed the KNN algorithm is a good choice as fitting a linear decision boundary would cause a lot of false classifications in the mixed area.

Given this choice to use KNN to be a good one, lets continue with the actual Machine Learning part. For this the GUI is ditched since it does not really add any value. Recall from the section [*The global idea of Machine Learning*](#The-global-idea-of-machine-learning) that in machine learning there are 2 key parts: Prediction and Validation. First we will look at the Validation, as using a model without any validation is never a good idea. The main reason to validate the model here is to prevent [overfitting](#overfitting). However, before we even can do validation, a *correct* K should be chosen. 

The drawback is that there is no golden rule for finding the correct K. However finding a K that allows for most datapoints to be classified correctly can be done by looking at the data. Additionally The K should be picked carefully to prevent undecidability by the algorithm. Say for example ```K=2```, and the problem has 2 labels, then when a point is between both labels, which one should the algorithm pick. There is a *rule of Thumb* that K should be the square root of the number of features (on other words the number of dimensions). In our example this would be ```K=1```, but this is not really a good idea since this would lead to higher false-classifications around decision boundaries. Picking ```K=2``` would result in the error regarding our two labels, thus picking ```K=3``` seems like a good fit for now.

For this example we do [2-fold Cross Validation](http://en.wikipedia.org/wiki/Cross-validation_(statistics) ). In general 2-fold Cross validation is a rather weak method of model Validation, as it splits the dataset in half and only validates twice, which still allows for overfitting, but since the dataset is only 100 points, 10-fold (which is a stronger version) does not make sense, since then there would only be 10 datapoints used for testing, which would give a skewed error rate.

```scala

  def main(args: Array[String]): Unit = {
   val basePath = "/.../KNN_Example_1.csv"
    val testData = GetDataFromCSV(new File(basePath))

    //Define the amount of rounds, in our case 2 and initialise the cross validation
    val validationRounds = 2;
    val cv = new CrossValidation(testData._2.length, validationRounds)
    //Then for each round
    for (i <- 0 to validationRounds - 1) {

      //Generate a subset of data points and their classifiers for Training
      val dpForTraining = testData._1.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)
      val classifiersForTraining = testData._2.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)

      //And the corresponding subset of datapoints and their classifiers for testing
      val dpForTesting = testData._1.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)
      val classifiersForTesting = testData._2.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)

      //Then generate a Model with KNN with K = 3 
      val knn = KNN.learn(dpForTraining, classifiersForTraining, 3)

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
If you execute this code several times you might notice the false prediction rate to fluctuate quite a bit. This is due to the random samples taken for training and testing. When this random sample is taken a bit unfortunate, the error rate becomes much higher while when taking a good random sample, the error rate could be extremely low. 

Unfortunately I cannot provide you with a golden rule to when your model was trained with the best possible training set. One would say the model with the least error rate is always the best, but when you recall the term [overfitting](#overfitting) picking this particular model might perform really bad on future data. This is why having a large enough and representative dataset is key to a good Machine Learning application. However when aware of this issue, you could implement manners to keep updating your model based on new data and known correct classifications.


Let's suppose you implement the KNN into your application, then you should have gone through the following steps. First you took care of getting the training and testing data. Next up you generated and validated several models and picked the model which gave the best results. After these steps you can finally do predictions using your ML implementations:


```scala
 
val unknownDataPoint = Array(5.3, 4.3)
val result = knn.predict(unknownDatapoint)
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

```

These predictions can then be used to present to the users of your system, for example as friend suggestion on a social networking site. The feedback the user gives on these predictions is valuable and should thus be fed into the system for updating your model.
 


###Classifying Email as Spam or Ham (Naive Bayes)
The goal of this section is to use the Naive Bayes implementation from [Smile](https://github.com/haifengl/smile) in Scala to classify emails as Spam or Ham based on their content.  

To start with this example I assume you created a new Scala project in your favourite IDE, and downloaded and added the [Smile Machine learning](https://github.com/haifengl/smile/releases)  and its dependency [SwingX](https://java.net/downloads/swingx/releases/) to this project. As final assumption you also downloaded and extracted the [example data](./Example%20Data/NaiveBayes_Example_1.zip). This example data comes from the [SpamAssasins public corpus](http://spamassasin.apache.org/publiccorpus/). 

As with every machine learning implementation, the first step is to load in the training data. However in this example we are taking it 1 step further into machine learning. In the [KNN examples](#Labeling ISPs based on their Down/Upload speed (K-NN using Smile in Scala)) we had the download and upload speed as [features](#features). We did not refer to them as features, as they where the only properties available. For spam classification it is not completely trivial what to use as features. One can use the Sender, the subject, the message content, and even the time of sending as features for classifying as spam or ham.  

In this example we will use the content of the email as feature. By this we mean, we will select the features (words in this case) from the bodies of the emails in the training set. In order to be able to do this, we need to build a [Term Document Matrix (TDM)](http://en.wikipedia.org/wiki/Document-term_matrix). We could use a library for this, but in order to gain more insight, let's build it ourselves. This also gives us all freedom in changing around stuff for properly selecting features.


We will start off with writing the functions for loading the example data. This will be done with a ```getMessage``` method which gets a filtered body from an email given a filename as parameter.

```scala

def getMessage(file : File)  : String  =
  {
    //Note that the encoding of the example files is latin1, thus this should be passed to the from file method.
    val source = scala.io.Source.fromFile(file)("latin1")
    val lines = source.getLines mkString "\n"
    source.close()
    //Find the first line break in the email, as this indicates the message body
    val firstLineBreak = lines.indexOf("\n\n")
    //Return the message body filtered by only text from a-z and to lower case 
    return lines.substring(firstLineBreak).replace("\n"," ").replaceAll("[^a-zA-Z ]","").toLowerCase()
  }
```

Now we need a method that gets all the filenames for the emails, from the example data folder structure that we provided you with.

```scala
  
  def getFilesFromDir(path: String):List[File] = {
    val d = new File(path)
    if (d.exists && d.isDirectory) {
      //Remove the mac os basic storage file, and alternatively for unix systems "cmds"
      d.listFiles.filter(_.isFile).toList.filter(x => ! x.toString.contains(".DS_Store") && ! x.toString.contains("cmds"))
    } 
    else {
      List[File]()
    }
  }
```

And finally lets define a set of paths that make it easier to load the different datasets from the example data. Together with this we also directly define a sample size of 500, as this is the complete amount of training emails are available for the spam set. We take the same amount of ham emails as the training set should be balanced for these two classification groups.

```scala
  
  def main(args: Array[String]): Unit = {
    val basePath = "/Users/../Downloads/data"
    val spamPath = basePath + "/spam"
    val spam2Path = basePath + "/spam_2"
    val easyHamPath = basePath + "/easy_ham"
    val easyHam2Path = basePath + "/easy_ham_2"
    val hardHamPath = basePath + "/hard_ham"
    val hardHam2Path = basePath + "/hard_ham_2"

  	val amountOfSamplesPerSet = 500;
    val amountOfFeaturesToTake = 100;    
    //First get a subset of the filenames for the spam sample set (500 is the complete set in this case)
    val listOfSpamFiles =   getFilesFromDir(spamPath).take(amountOfSamplesPerSet)
    //Then get the messages that are contained in these files
    val spamMails = listOfSpamFiles.map{x => (x,getMessage(x)) }
    
     //Get a subset of the filenames from the ham sampleset (note that in this case it is not necessary to randomly sample as the emails are already randomly ordered)
  	val listOfHamFiles =   getFilesFromDir(easyHamPath).take(amountOfSamplesPerSet)
  	//Get the messages that are contained in the ham files
  	val hamMails  = listOfHamFiles.map{x => (x,getMessage(x)) }
  }

  
```


Now that we have the training data for both the ham and the spam email, we can start building 2 [TDM's](http://en.wikipedia.org/wiki/Document-term_matrix). But before we show you the code for this, lets first explain shortly why we actually need this. The TDM will contain **ALL** words which are contained in the bodies of the training set, including frequency rate. However, since frequency might not be the best measure (as 1 email which contains 1.000.000 times the word 'pie' would mess up the complete table) we will also compute the **occurrence rate**. By this we mean, the amount of documents that contain that specific term. So lets start off with implementing the TDM.


```scala

class TDM {

  var records : List[TDMRecord] =  List[TDMRecord]()

  def addTermToRecord(term : String, documentName : String)  =
    {
      //Find a record for the term
      val record =   records.find( x => x.term == term)
      if (record.nonEmpty)
      {
        val termRecord =  record.get
        val documentRecord = termRecord.occurrences.find(x => x._1 == documentName)
        if (documentRecord.nonEmpty)
        {
           termRecord.occurrences +=  documentName -> (documentRecord.get._2 + 1)
        }
        else
        {
          termRecord.occurrences +=  documentName ->  1
        }
      }
      else
      {
        //No record yet exists for this term
        val newRecord  = new TDMRecord(term, mutable.HashMap[String,Int](documentName ->  1))
        records  = newRecord :: records
      }
    }
  def SortByTotalFrequency = records = records.sortBy( x => -x.totalFrequency)
  def SortByOccurrenceRate(rate : Int) = records = records.sortBy( x => -x.occurrenceRate(rate))
}

class TDMRecord(val term : String, var occurrences :  mutable.HashMap[String,Int] )
{
  def totalFrequency = occurrences.map(y => y._2).fold(0){ (z, i) => z + i}
  def occurrenceRate(totalDocuments : Int) : Double  = occurrences.size.toDouble / totalDocuments
  def densityRate(totalTerms : Int) : Double  = totalFrequency.toDouble / totalTerms;
}
```

As you can see there are two sort methods: ```SortByTotalFrequency``` and ```SortByOccurenceRate```. In the latter one you need to pass the rate, which represents the total amount of documents that are contained in the TDM. This is done for performance reasons, since the TMD does not keep track of the amount of documents that was used to build up this TDM. Given this implementation, we can now actually make the two tables, one for spam and one for ham. We will add this code to the main class.

```scala

val spamTDM = new TDM();
//Build up the Term-Document Matrix for spam emails
spamMails.foreach(x => x._2.split(" ").filter(_.nonEmpty).foreach(y => spamTDM.addTermToRecord(y,x._1.getName))
//Sort the spam by the occurrence rate to gain more insight
spamTDM.SortByOccurrenceRate(hamMails.size)
 
val hamTDM = new TDM();
//Build up the Term-Document Matrix for ham emails
hamMails.foreach(x => x._2.split(" ").filter(_.nonEmpty).foreach(y => hamTDM.addTermToRecord(y,x._1.getName)))
//Sort the ham by the occurrence rate to gain more insight
hamTDM.SortByOccurrenceRate(spamMails.size)


```

Given the tables, lets take a look at the top 50 words for each table. Note that the red words are from the spam table and the green words are from the ham table. Additionally, the size of the words represents the occurrence rate. Thus the larger the word, the more documents contained that word at least once.

<img src="./Images/Ham_Stopwords.png" width="400px" height="200px" />
<img src="./Images/Spam_Stopwords.png" width="400px" height="200px" />

As you can see, mostly stop words come forward. These stopwords are noise, which we should not use in our feature selection, this we should remove these from the tables before selecting the features. We've included a list of stopwords in the example dataset. Lets first define the code to get these stopwords.
```scala
  def getStopWords() : List[String] =
  {
    val source = scala.io.Source.fromFile(new File("/Users/.../.../Example Data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    return  lines.toList
  }
 
```

Now we can expand the main body with removing the stopwords from the Tables

```scala
//Filter out all stopwords
hamTDM.records = hamTDM.records.filter(x => !StopWords.contains(x.term));
spamTDM.records = spamTDM.records.filter(x => !StopWords.contains(x.term));

```

If we once again look at the top 50 words for spam and ham, we see that most of the stopwords are gone. We could fine-tune more, but for now lets go with this.

<img src="./Images/Ham_No_Stopwords.png" width="400px" height="200px" />
<img src="./Images/Spam_No_Stopwords.png" width="400px" height="200px" />

With this insight in what 'spammy' words and what typical 'ham-words' are, we can decide on building a feature-set which we can then use in the Naive Bayes algorithm for creating the classifier. Note: it is always better to include **more** features, however performance might become an issue when having all words as features. This is why in the field, developers tend to drop features that do not have a significant impact, purely for performance reasons. Alternatively machine learning is done running complete [Hadoop](http://hadoop.apache.org/) clusters, but explaining this would be outside the scope of this blog.

For now we will select the top 100 spammy words based on occurrence(thus not frequency) and do the same for ham words and combine this into 1 set of words which we can feed into the bayes algorithm. Finally we also convert the training data to fit the input of the Bayes algorithm. Note that the final feature set thus is 200 - (#intersecting words *2). Feel free to experiment with higher and lower feature counts.


```scala

//Add the code for getting the TDM data and combining it into a feature bag.
val hamFeatures = hamTDM.records.take(amountOfFeaturesToTake).map(x => x.term)
val spamFeatures = spamTDM.records.take(amountOfFeaturesToTake).map(x => x.term)

//Now we have a set of ham and spam features, we group them and then remove the intersecting features, as these are noise.
var data = (hamFeatures ++ spamFeatures).toSet
hamFeatures.intersect(spamFeatures).foreach(x => data = (data - x))

//Initialise a bag of words that takes the top x features from both spam and ham and combines them
var bag = new Bag[String] (data.toArray);
//Initialise the classifier array with first a set of 0(spam) and then a set of 1(ham) values that represent the emails
var classifiers =  Array.fill[Int](amountOfSamplesPerSet)(0) ++  Array.fill[Int](amountOfSamplesPerSet)(1)

//Get the trainingData in the right format for the spam mails
var spamData = spamMails.map(x => bag.feature(x._2.split(" "))).toArray

//Get the trainingData in the right format for the ham mails
var hamData = hamMails.map(x => bag.feature(x._2.split(" "))).toArray

//Combine the training data from both categories
var trainingData = spamData ++ hamData
```

Given this feature bag, and a set of training data, we can start training the algorithm. For this we can chose a few different models: 'general', 'multinomial' and Bernoulli. In this example we focus on the multinomial but feel free to try out the other model types as well.


```scala
//Create the bayes model as a multinomial with 2 classification groups and the amount of features passed in the constructor.
  var bayes = new NaiveBayes(NaiveBayes.Model.MULTINOMIAL, 2, data.size)
  //Now train the bayes instance with the training data, which is represented in a specific format due to the bag.feature method, and the known classifiers.

  bayes.learn(trainingData, classifiers)
```

Now that we have the trained model, we can once again do some validation. However, in the example data we already made a separation between easy and hard ham, and spam, thus we will not apply the cross validation, but rather validate the model these test sets. We will start with validation of spam classification. For this we use the 1397 spam emails from the spam2 folder.

```scala
val listOfSpam2Files =   getFilesFromDir(spam2Path)
val spam2Mails = listOfSpam2Files.map{x => (x,getMessage(x)) }
val spam2FeatureVectors = spam2Mails.map(x => bag.feature(x._2.split(" ")))
val spam2ClassificationResults = spam2FeatureVectors.map(x => bayes.predict(x))

//Correct classifications are those who resulted in a spam classification (0)
val correctClassifications = spam2ClassificationResults.count( x=> x == 0);
println(correctClassifications + " of " + listOfSpam2Files.length + "were correctly classified")
println(((correctClassifications.toDouble /  listOfSpam2Files.length) * 100)  + "% was correctly classified")

//In case the algorithm could not decide which category the email belongs to, it gives a -1 (unknown) rather than a 0 (spam) or 1 (ham)
val unknownClassifications = spam2ClassificationResults.count( x=> x == -1);
println(unknownClassifications + " of " + listOfSpam2Files.length + "were unknowingly classified")
println(((unknownClassifications.toDouble /  listOfSpam2Files.length) * 100)  + "% was unknowingly classified")

```

If we run this code several times with different feature amounts we get the following results:


| amountOfFeaturesToTake	| Spam (Correct)| Unknown| Ham | 
| ---------------------		|:-------------	| :----- |:----|
| 50      					| 1281 (91.70%)	| 16 (1.15%)	| 100 (7.15%) |
| 100     					| 1189 (85.11%)	| 18 (1.29%)	| 190 (13.6%)|
| 200     					| 1197 (85.68%)	| 16 (1.15%)	| 184 (13.17%)|
| 400     					| 1219 (87.26%)	| 13 (0.93%)	| 165 (11.81%)|

Interestingly enough, the algorithm works best with only 50 features. However, if you recall that there were still *stop words* in the top 50 classification terms which could explain this result.  If you look at how the values change as the amount of features increase (starting at 100), you can see that with more features, the overall result increases. Note that there are a group of unknown emails. For these emails the [prior](#Prior) was equal for both classes. Note that this also is the case if there are no feature words for ham nor spam in the email, because then the algorithm would classify it 50% ham 50% spam.


If we change the path to ```easyHam2Path``` and rerun the code for we get the following results:

| amountOfFeaturesToTake	| Spam | Unknown| Ham  (Correct) | 
| ---------------------		|:-------------	| :----- |:----|
| 50      					| 120 (8.57%)	| 28 ( 2.0%)	| 1252 (89.43%)	|
| 100   					| 44 (3.14%)	| 11 (0.79%)	| 1345 (96.07%)	|
| 200 						| 36 (2.57%)	| 7 (0.5%)	| 1357 (96.93%)	|
| 400     					| 24 (1.71%)	| 7 (0.5%)	| 1369 (97.79%) |

Here we see that indeed, when you use only 50 features, the amount of ham that gets classified correctly is significantly lower in comparison to the correct classifications when using 100 features.

We could work through the hard ham, but since the building bricks are already here, we leave this to the reader. 

###Predicting weight based on height (using Ordinary Least Squares)
In this section we will introduce the [Ordinary Least Squares](http://en.wikipedia.org/wiki/Ordinary_least_squares) technique which is a form of linear regression. As this technique is quite powerful, it is important to have read [regression](#regression) and the common pitfalls before starting with this example. We will cover some of these issues in this section, while others are shown in the sections [under-fitting](#under-fitting) and [overfitting](#overfitting)


As always, the first thing to do is to import a dataset. For this we provide you with the following [csv file](./Example%20Data/OLS_Regression_Example_3.csv) and code for reading this file:


```scala

  def GetDataFromCSV(file: File): (Array[Array[Double]], Array[Double]) = {
    val source = scala.io.Source.fromFile(file)
    val data = source.getLines().drop(1).map(x => GetDataFromString(x)).toArray
    source.close()
    var inputData = data.map(x => x._1)
    var resultData = data.map(x => x._2)

    return (inputData,resultData)
  }

  def GetDataFromString(dataString: String): (Array[Double], Double) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')
    var person = 1.0;

    if (dataArray(0) == "\"Male\"") {
      person = 0.0
    }

    //Extract the values from the strings
    //Since the data is in US metrics (inch and pounds we will recalculate this to cm and kilo's)
    val data : Array[Double] = Array(person,dataArray(1).toDouble * 2.54)
    val weight: Double = dataArray(2).toDouble * 0.45359237

    //And return the result in a format that can later easily be used to feed to Smile
    return (data, weight)
  }

```

Note that the data reader converts the values from the [Imperial system](http://en.wikipedia.org/wiki/Imperial_units) into the [Metric] system(http://en.wikipedia.org/wiki/Metric_system). This has no big effects on the OLS implementation, but since the metric system is more commonly used, we prefer to present the data in that system.

With these methods we get the data as an ```Array[Array[Double]]``` which represents the datapoints and an ```Array[Double]``` which represents the classifications as male or female. These formats are good for both plotting the data, and for feeding into the machine learning algorithm.

Let's first see what the data looks like. For this we plot the data using the following code.


```scala

object LinearRegressionExample extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "Linear Regression Example"
    val basePath = "/Users/.../OLS_Regression_Example_3.csv"

    val test_data = GetDataFromCSV(new File(basePath))

    val plotData = (test_data._1 zip test_data._2).map(x => Array(x._1(1) ,x._2))
    val maleFemaleLabels = test_data._1.map( x=> x(0).toInt);
    val plot =  ScatterPlot.plot(plotData,maleFemaleLabels,'@',Array(Color.blue, Color.green))
    plot.setTitle("Weight and heights for male and females")
    plot.setAxisLabel(0,"Heights")
    plot.setAxisLabel(1,"Weights")



    peer.setContentPane(plot)
    size = new Dimension(400, 400)
  }

```

If you execute the code here above, a window will pop up which shows the **right** plot as shown in the image here below. Note that when you run the code, you can scroll to zoom in and out in the plot.

<img src="./Images/HumanDataPoints.png" width="275px" />
<img src="./Images/MaleFemalePlot.png" width="275px" />


In this plot, given that green is female, and blue is male, you can see that there is a big overlap in their weights and heights. So if we where to ignore the difference between male and female it would still look like there was a linear function in the data (which can be seen in the **left** plot). However when ignoring this difference, the function would be not as accurate as it would be when we incorporate the information regarding males and females. 

Finding this distinction is trivial in this example, but you might encounter datasets where these groups are not so obvious. Making you aware of this this possibility might help you find groups in your data, which can improve the performance of your machine learning application.

Now that we have seen the data and see that indeed we can come up with a linear regression line to fit this data, it is time to train a [model](#Model). Smile provides us with the [ordinary least squares](http://en.wikipedia.org/wiki/Ordinary_least_squares) algorithm which we can easily use as follows:

```scala
val olsModel = new OLS(test_data._1,test_data._2)
```

With this olsModel we can now predict someone's weight based on length and gender as follows: 

```scala
println("Prediction for Male of 1.7M: " +olsModel.predict(Array(0.0,170.0)))
println("Prediction for Female of 1.7M:" + olsModel.predict(Array(1.0,170.0)))
println("Model Error:" + olsModel.error())
```

and this will give the following results:

```
Prediction for Male of 1.7M: 79.14538559840447
Prediction for Female of 1.7M:70.35580395758966
Model Error:4.5423150758157185
```

If you recall from the classification algorithms, there was a [prior](#Prior) value to be able to say something about the performance of your model. Since regression is a stronger statistical method, you have an actual error value now. This value represents how far off the fitted regression line is in average, such that you can say that for this model, the prediction for a male of 1.70m is 79.15kg  +/- the error of 4.54kg. Note that if you would remove the distinction between males and females, this error would increase to 5.5428. In other words, adding the distinction between male and female, increases the model accuracy by +/- 1 kg in its predictions.

Finally smile also provides you with some statistical information regarding your model. The method ```RSquared``` gives you the [root-mean-square error (RMSE)](#Root Mean Squared Error (RMSE)) from the model divided by the [RMSE](#Root Mean Squared Error (RMSE)) from the mean. This value will always be between 0 and 1. If your model predicts every datapoint perfectly, RSquared will be 1, and if the model does not perform better than the mean function, the value will be 0. In the field this measure is often multiplied by 100 and then used as representation of how accurate the model is. Because this is a normalised value, it can be used to compare the performance of different models.

This concludes linear regression, if you want to know more about how to apply regression on  non-linear data, feel free to work through the next example [Predicting O'Reilly top 100 selling books using text regression](#Predicting O'Reilly top 100 selling books using text regression).



###Predicting O'Reilly top 100 selling books using text regression

In the example of [predicting weights based on heights and gender](#Predicting-weight-based-on height-(using-Ordinary-Least-Squares)) we introduced the notion of linear regression. However, sometimes one would want to apply regression on non numeric data such as Text.

In this example we will show how to do this, but this example will also show that for this particular case it does not work out to use text regression. The reason for this is that the data simply does not contain a signal for our test data. However this does not make this example useless because there might be an actual signal within the text in the data you are using in practice,which then can be detected using text regression as explained here.


Let's start off with getting the data we need:

```scala
object TextRegression  {

  def main(args: Array[String]): Unit = {

    //Get the example data
      val basePath = "/users/.../TextRegression_Example_4.csv"
      val testData = GetDataFromCSV(new File(basePath))
  }

  def GetDataFromCSV(file: File) : List[(String,Int,String)]= {
    val reader = CSVReader.open(file)
    val data = reader.all()

    val documents = data.drop(1).map(x => (x(1),x(3)toInt,x(4)))
    return documents
  }
}

```

We now have the title, rank and long description of the top 100 selling books from O'Reilly. However when we want to do regression of some form, we need numeric data. This is why we will build a [Document Term Matrix (DTM)](http://en.wikipedia.org/wiki/Document-term_matrix). Note that this DTM is similar to the Term Document Matrix (TDM) that we built in the spam classification example. It's difference is that we store document records containing which terms are in that document, in contrast to the TDM where we store records of words, containing a list of documents in which this term is available.

We implemented the DTM ourselves as follows:

```scala
import java.io.File
import scala.collection.mutable

class DTM {

  var records: List[DTMRecord] = List[DTMRecord]()
  var wordList: List[String] = List[String]()

  def addDocumentToRecords(documentName: String, rank: Int, documentContent: String) = {
    //Find a record for the document
    val record = records.find(x => x.document == documentName)
    if (record.nonEmpty) {
      throw new Exception("Document already exists in the records")
    }

    var wordRecords = mutable.HashMap[String, Int]()
    val individualWords = documentContent.toLowerCase.split(" ")
    individualWords.foreach { x =>
      val wordRecord = wordRecords.find(y => y._1 == x)
      if (wordRecord.nonEmpty) {
        wordRecords += x -> (wordRecord.get._2 + 1)
      }
      else {
        wordRecords += x -> 1
        wordList = x :: wordList
      }
    }
    records = new DTMRecord(documentName, rank, wordRecords) :: records
  }

  def getStopWords(): List[String] = {
    val source = scala.io.Source.fromFile(new File("/Users/.../stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    return lines.toList
  }

  def getNumericRepresentationForRecords(): (Array[Array[Double]], Array[Double]) = {
    //First filter out all stop words:
    val StopWords = getStopWords()
    wordList = wordList.filter(x => !StopWords.contains(x))
    
    var dtmNumeric = Array[Array[Double]]()
    var ranks = Array[Double]()

    records.foreach { x =>
      //Add the rank to the array of ranks
      ranks = ranks :+ x.rank.toDouble

      //And create an array representing all words and their occurrences for this document:
      var dtmNumericRecord: Array[Double] = Array()
      wordList.foreach { y =>

        val termRecord = x.occurrences.find(z => z._1 == y)
        if (termRecord.nonEmpty) {
          dtmNumericRecord = dtmNumericRecord :+ termRecord.get._2.toDouble
        }
        else {
          dtmNumericRecord = dtmNumericRecord :+ 0.0
        }
      }
      dtmNumeric = dtmNumeric :+ dtmNumericRecord

    }

    return (dtmNumeric, ranks)
  }
}

class DTMRecord(val document : String, val rank : Int, var occurrences :  mutable.HashMap[String,Int] )


```

If you look at this implementation you'll notice that there is a method called  ```def getNumericRepresentationForRecords(): (Array[Array[Double]], Array[Double])```. This method returns a tuple with as first parameter a matrix in which each row represents a document, and each column represents one of the words from the complete vocabulary of the DTM's documents. Note that the doubles in the first table represent the # of occurrences of the words.

The second parameter is an array containing all the ranks beloning to the records from the first table. 

We can now extend our main code such that we get the numeric representation of all the documents as follows:

```scala

val documentTermMatrix  = new DTM()
testData.foreach(x => documentTermMatrix.addDocumentToRecords(x._1,x._2,x._3))

```


With this conversion from text to numeric value's we can open our regression toolbox. We used [Ordinary Least Squares(OLS)](http://en.wikipedia.org/wiki/Ordinary_least_squares) in the example [predicting weight based on height](###Predicting-weight-based-on-height-(using-Ordinary-Least-Squares)), however this time we will use [Least Absolute Shrinkage and Selection Operator (Lasso)](http://en.wikipedia.org/wiki/Least_squares#Lasso_method) regression. This is because we can give this regression method a certain lambda, which represents a penalty value. This penalty value allows the LASSO algorithm to select relevant features while discarding some of the other features. 

This feature selection that Lasso performs is very useful in our case due too the large set of words that is used in the documents descriptions. Lasso will try to come up with an ideal subset of those words as features, where as when applying the OLS, all words would be used, and the runtime would be extremely high. Additionally, the OLS implementation of SMILE detects rank deficiency. This is one of the [curses of dimensionality](#Curse-of-dimensionality)

We need to find an optimal lambda however, thus we should try for several lambda using cross validation. We will do this as follows:

```scala

    for (i <- 0 until cv.k) {
      //Split off the training datapoints and classifiers from the dataset
      val dpForTraining = numericDTM._1.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)
      val classifiersForTraining = numericDTM._2.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)

      //And the corresponding subset of data points and their classifiers for testing
      val dpForTesting = numericDTM._1.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)
      val classifiersForTesting = numericDTM._2.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)

      //These are the lambda values we will verify against
      val lambdas: Array[Double] = Array(0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

      lambdas.foreach { x =>
        //Define a new model based on the training data and one of the lambda's
        val model = new LASSO(dpForTraining, classifiersForTraining, x)

        //Compute the RMSE for this model with this lambda
        val results = dpForTesting.map(y => model.predict(y)) zip classifiersForTesting
        val RMSE = Math.sqrt(results.map(x => Math.pow(x._1 - x._2, 2)).sum / results.length)
        println("Lambda: " + x + " RMSE: " + RMSE)

      }
    }
    
```

Running this code multiple times gives an RMSE varying between 36 and 51. This means that the rank prediction we would do would be off by at least 36 ranks. Given the fact that we are trying to predict the top 100 ranks, it shows that the algorithm performs very poorly. Differences in lambda are in this case not noticeable. However when using this in practice you should be careful when picking the lambda value: The higher the lambda you pick, the lower the amount of features for the algorithm becomes. This is why cross validation is important to see how the algorithm performs on different lambda's.

To conclude this example, we rephrase a quote from [John Tukey](http://en.wikipedia.org/wiki/John_Tukey): 
>The data may not contain the answer. The combination of some data and an aching desire for an answer does not ensure that a reasonable answer can be extracted from a given body of data.



###Using unsupervised learning to create a market index (PCA)

//Todo: write

###using Support Vector Machine's

//Todo: write

##The global idea of machine learning
The term 'Machine learning' is known by almost everyone, however almost no-one I spoke could really explain what it is in **_one_** or even several sentences.

*Explanation of machine learning, preferably with an graphical image*


In the upcoming subsections the most important notions you need to be aware off when practicing machine learning are (briefly) explained.

###Features
A feature (in the field of machine learning) is a property on which a [Model](#Model) is trained. Say for example that you classify emails as spam/ham based on the frequency of the word 'Buy' and 'Money'. Then these words are features. If you would use machine learning to predict whether one is a friend of you, the amount of 'common' friends could be a feature.

###Model
When one talks about machine learning, often the term *model* is mentioned. The model is the result of any machine learning method and the algorithm used within this method. This model can be used to make predictions in [supervised](#Supervised Learning), or to retrieve clusterings in [unsupervised learning](#Unsupervised learning).

###Learning methods
In the field of machine learning there are two leading ways of learning, namely [Supervised learning](http://en.wikipedia.org/wiki/Supervised_learning) and  [Unsupervised learning](http://en.wikipedia.org/wiki/Unsupervised_learning). A brief introduction is necessary when you want to use Machine learning in your applications, as picking the right machine learning approach is an important but sometimes also a little tedious process.

####Supervised Learning
The principle of supervised learning can be used to solve many problem types. In this blog however we will stick to [Classification](#Classification) and [Regression](#Regression) as this covers most of the problems one wants to solve in their every day application.


#####Classification
The problem of classification within the domain of Supervised learning is relatively simple. Given a set of labels, and some data that already received the correct labels, we want to be able to *predict* labels for new data that we did not label yet. However, before thinking of your data as a classification problem, you should look at what the data looks like. If there is a clear structure in the data such that you can easily draw a regression line it might be better to use a [regression](#Regression) algorithm instead. Given the data does not fit to a regression line, or when performance becomes an issue, classification is a good alternative.

An example of a classification problem would be to classify emails as Ham or Spam based on their content. Given a training set in which emails are labeled Ham/Spam, a classification algorithm can be used to train a [Model](#Model). This model can then be used to predict for future emails whether they are Ham or Spam. A typical example of a classification algorithm is the [K-NN algorithm](#Labeling ISPs based on their Down/Upload speed (K-NN using Smile in Scala)). Another more commonly used example of a classification problem is [Classifying Email as Spam or Ham](###Classifying Email as Spam or Ham (Naive Bayes)) which is also one of the examples written on this blog.

#####Regression
Regression is a lot stronger in comparison to [classification](#classification). This is because in regression you are predicting actual values, rather than labels. Let us clarify this with a short example: given a table of weights, heights, and genders, you can use [KNN](#Labeling ISPs based on their Down/Upload speed (K-NN using Smile in Scala)) to predict ones gender when given a weight and height. With this same dataset using regression, you could instead predict ones weight or height, given the gender the respective other missing parameter. 

With this extra power, comes great responsibility, thus in the working field of regression one should be very careful when generating the model. Common pitfalls are [overfitting](#overfitting), [under fitting](#under-fitting) and not taking into account how the model handles  [extrapolation](http://en.wikipedia.org/wiki/Extrapolation) and [interpolation](http://en.wikipedia.org/wiki/Interpolation).



####Unsupervised Learning


#####Principal Components Analysis (PCA)
Principal Components Analysis is a technique used in statistics to convert a set of correlated columns into a smaller set of uncorrelated columns, reducing the amount features of a problem.  This smaller set of columns are called the principal components. This technique is mostly used in exploratory data analysis as it reveals internal structure in the data that can not be found with eye-balling the data.

A big weakness of PCA however are outliers in the data. these heavily influence it's result, thus looking at the data on beforehand, eliminating large outliers can greatly improve its performance.


###Validation techniques
In this section we will explain some of the techniques available for model validation, and will explain some terms that are commonly used in the Machine Learning field.


#### Cross Validation
The technique of cross validation is one of the most common techniques in the field of machine learning. It's essence is to *ignore* part of your dataset while training your [model](#model), and then using the model to predict this *ignored data*. Comparing the predictions to the actual value then gives an indication of the performance of your model.


#####(2 fold) Cross Validation

#### Regularization
The basic idea of regularization is preventing [overfitting](#overfitting) your [model](#model) by simplifying it. Suppose your data is a polynomial function of degree 3, but your data has noise and this would cause the model to be of a higher degree. Then the model would perform poorly on new data, where as it seems to be a good model at first. Regularization hels preventing this, by simplifying the model with a certain value *lambda*. However to find the right lambda for a model is hard when you have no idea as to when the model is overfitted or not. This is why [cross validation](#Cross-Validation) is often used to find the best lambda fitting your model.


##### Precision

##### Recall

##### Prior
The prior value that belongs to a classifier given a datapoint represents the likelihood that this datapoint belongs to this classifier. 

##### Root Mean Squared Error (RMSE)
The Root Mean Squared Error (RMSE or RMSD where D is deviation) is the square root of the variance of the differences between the actual value and predicted value.
Suppose we have the following values:

| Predicted temperature | Actual temperature |  squared difference for Model | square difference for average |
| :--: | :--:| :--:| :--: | 
|10 | 12 | 4 |  7.1111 |
|20 | 17 | 9 |  5.4444 |
|15 | 15 | 0 |  0.1111 |

Then the mean of this squared difference for the model is 4.33333, and the root of this is 2.081666. So basically in average, the model predicts the values with an average error of 2.09. The lower this RMSE value is, the better the model is in its predictions. This is why in the field, when selecting features, one computes the RMSE with and without a certain feature, in order to say something about how that feature affects the performance of the model.


Additionally, because the RMSE is an absolute value, it can be normalised in order to compare models. This results in the Normalised Root Mean Square Error (NRMSE). For computing this however, you need to know the minimum and maximum value that the system can contain. Let's suppose we can have temperatures ranging from minimum of 5 to a maximum of 25 degrees, then computing the NRMSE is as follows:

<img src="./Images/Formula4.png"/>

When we fill in the actual values we get the following result:

<img src="./Images/Formula1.png"/>

Now what is this 10.45 value?  This is the error percentage the model has in average on it's datapoints.

Finally we can use RMSE to compute a value that is known in the field as **R Squared**. This value represents how good the model performs in comparison to ignoring the model and just taking the average for each value. For that you need to calculate the RMSE for the average first. This is 4.22222  (taking the mean of the values from the last column in the table), and the root is then 2.054805. The first thing you should notice is that this value is lower than that of the model. This is not a good sign, because this means the model performs **worse** than just taking the mean. However to demonstrate how to compute **R Squared** we will continue the computations.

We now have the RMSE for both the model and the mean, and then computing how well the model performs in comparison to the mean is done as follows:

<img src="./Images/Formula2.png"  />

This results in the following computation:

<img src="./Images/Formula3.png"  />

Now what does this -1.307229 represent. Basically it says that the model that predicted these values performs +/- 1.31 percent worse than returning the average each time a value is to be predicted. In other words, we could better use the average function as a predictor rather than the model in this specific case.


###Pitfalls 
This section describes some common pitfalls in applying machine learning techniques. The idea of this section is to make you aware of of these pitfalls and help you prevent actually walking into one yourself.

##### Overfitting

When fitting a function on the data, there is a possibility the data contains noise (for example by measurement errors). If you fit every point from the data exactly, you incorporate this noise into the [model](#model). This causes the model to predict really well on your test data, but relatively poor on future data.

The left image here below show how this overfitting would look like if you where to plot your data and the fitted functions, where as the right image would represent a *good fit* of the regression line through the datapoints.


<img src="./Images/OverFitting.png" width="300px" /> 
<img src="./Images/Good_Fit.png" width="300px" />

Overfitting can easily happen when applying [regression](#regression) but can just as easily be introduced in [nbayes classifications](#Classifying Email as Spam or Ham (Naive Bayes)). In regression it happens with rounding, bad measurements and noisy data. In naive bayes however, it could be the features that where picked. An example for this would be classifying spam or ham while keeping all stopwords.

Overfitting can be detected by performing [validation techniques](#Validation-techniques) and looking into your data's statistical features, and detecting and removing outliers.



##### Under-fitting
When you are turning your data into a model, but are leaving (a lot of) statistical data behind, this is called under-fitting. This can happen due to various reasons, such as using a wrong regression type on the data. If you have a non-linear structure in the data, and you apply linear regression, this would result in an under-fitted model. The left image here below represents a under-fitted regression line where as the right image shows a good fit regression line.

<img src="./Images/Under-fitting.png" width="300px" /> 
<img src="./Images/Good_Fit.png" width="300px" />

You can prevent under fitting by plotting the data to get insights in the underlying structure, and using [validation techniques](#validation-techniques) such as [cross validation](#(2-fold)-Cross-Validation). 

##### Curse of dimensionality
The curse of dimensionality is a collection of problems that can occur when your data size is lower than the amount of features (dimensions) you are trying to use to create your machine learning [model](#model). An example of a dimensionality curse is matrix rank deficiency. When using [Ordinary Least Squares(OLS)](http://en.wikipedia.org/wiki/Ordinary_least_squares), the underlying algorithm solves a linear system in order to build up a model. However if you have more columns than you have rows, solving this system is not possible. If this is the case, the best solution would be to get more datapoints or reduce the feature set. 

If you want to know more regarding this curse of dimensionality, [a study focussed on this issue](http://lectures.molgen.mpg.de/networkanalysis13/LDA_cancer_classif.pdf). In this study, researchers Haifeng Li, Keshu Zhang and Tao Jiang developed an algorithm that improves cancer classification with very few datapoints in comparison to [support vector machines](http://en.wikipedia.org/wiki/Support_vector_machine) and [random forests](http://en.wikipedia.org/wiki/Random_forest)
