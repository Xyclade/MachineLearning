#Machine Learning for developers 


Most developers these days have heard of machine learning, but when trying to find an 'easy' way into this technique, most people find themselves getting scared off by the abstractness of the concept of *Machine Learning* and terms as   [*regression*](http://en.wikipedia.org/wiki/Regression_analysis), [*unsupervised learning*](http://en.wikipedia.org/wiki/Unsupervised_learning), [*Probability Density Function*](http://en.wikipedia.org/wiki/Probability_density_function) and many other definitions. 

And then there are books such as [*Machine Learning for Hackers*](http://shop.oreilly.com/product/0636920018483.do) and [*An Introduction to Statistical Learning with Applications in R*](http://www-bcf.usc.edu/~gareth/ISL/)  who use programming language [R](http://www.r-project.org) for their examples. 

However R is not really a programming language in which one writes programs for everyday use such as is done with for example Java, C#, Scala etc. This is why in this blog, Machine learning will be introduced using libraries for java/scala and C# which are languages that almost every developer has worked with once during their study or carreer.

Note that in this blog, 'new' definitions are hyperlinked such that if you want, you **can** read more regarding that specific topic, but you are not obliged to do this in order to be able to work through the examples. However the section ['The global idea of machine learning'](#The global idea of machine learning) helps making things a lot more clear when working through the examples and is advised to be read on beforehand in case you are completely new to Machine Learning.




##Practical Examples

The examples will start off with the most simple and intuitive [*Classification*](http://en.wikipedia.org/wiki/Statistical_classification) Machine learning Algorithm there is: [*K-Nearest Neighbours*](http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

###Labeling ISPs based on their Down/Upload speed (K-NN using Smile in Scala)

The goal of this section is to use the K-NN implementation from [Smile](https://github.com/haifengl/smile) in Scala to classify download/upload speed pairs as [ISP](http://en.wikipedia.org/wiki/Internet_service_provider) Alpha or Beta.  

To start with this example I assume you created a new Scala project in your favorite IDE, and downloaded and added the [Smile Machine learning](https://github.com/haifengl/smile/releases) to this project. As final assumption you also downloaded the [example data](https://github.com/johnmyleswhite/ML_for_Hackers/blob/master/10-Recommendations/data/example_data.csv).

The first step is to load the CSV data file. As this is no rocket science, I provide the code for this without further explanation:

```scala
  def main(args: Array[String]): Unit = {
    val basePath = "/.../example_data.csv"
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
```

First thing you might wonder now is *why the frick is the data formatted that way*. Well, the separation between dataPoints and their label values is for easy splitting between testing and training data, and because the API expects the data this way. Secondly the datapoints stored as an ```scala Array[Array[Double]]``` is done to support datapoints in more than just 2 dimensions.

Given the data the first thing to do next is to see what the data looks like. For this Smile provides a nice 






##The global idea of machine learning
The term 'Machine learning' is known by almost everyone, however almost no-one I spoke could really explain what it is in **_one_** or even serveral sentences.

*Explanation of machine learning, preferably with an graphical image*


In the upcoming subsections the most important notions you need to be aware off when practicing machine learning are (briefly) explained.


####Model

####Validation techniques

##### Overfitting
##### Underfitting

#####Precision

#####Recall

####Exploratory Data Analysis








