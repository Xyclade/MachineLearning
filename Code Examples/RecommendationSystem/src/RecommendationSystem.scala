import java.awt.{Rectangle, ComponentOrientation}
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import smile.plot.BarPlot

import scala.swing.{MainFrame, SimpleSwingApplication}
import scala.util.Try

object RecommendationSystem extends SimpleSwingApplication {

  def top = new MainFrame {
    title = "Recommendation System Example"

    val basePath = "/Users/mikedewaard/ML_for_Hackers/03-Classification/data"
    val easyHamPath = basePath + "/easy_ham"
    val easyHam2Path = basePath + "/easy_ham_2"
    val hardHamPath = basePath + "/hard_ham"
    val hardHam2Path = basePath + "/hard_ham_2"


    val listOfSpamFiles = getFilesFromDir(easyHamPath)

    val mailBodies = listOfSpamFiles.map(x => (x,getFullEmail(x)))

    val mailInformation = mailBodies.map(x => (x._2, getDateFromEmail(x._2), getSenderFromEmail(x._2), getSubjectFromEmail(x._2), getMessageBodyFromEmail(x._2), x._1.toString))

    val timeSortedMails = mailInformation.sortBy(x => x._2)

    val testAndTrainingSplit = timeSortedMails.splitAt(timeSortedMails.length / 2)


    val trainingData = testAndTrainingSplit._1

    val testingData = testAndTrainingSplit._2

    val mailsGroupedBySender = trainingData.groupBy(x => x._3)
    val senderBarPlotData = mailsGroupedBySender.map(x => (x._1, x._2.length)).toArray.sortBy(x => x._2)
    val senderDescriptions = senderBarPlotData.map(x => x._1)
    val senderValues = senderBarPlotData.map(x => Math.log1p(x._2.toDouble))

    val mailsGroupedByThread = trainingData.groupBy(x => x._4)
    val mailGroupsWithMinMaxDates = mailsGroupedByThread.map(x => (x._1, x._2, (x._2.maxBy(x => x._2)._2.getTime - x._2.minBy(x => x._2)._2.getTime) / 1000))
    val threadGroupsWithWeights = mailGroupsWithMinMaxDates.filter(x => x._3 != 0).map(x => (x._1, x._2, x._3, 10 + Math.log10(x._2.length.toDouble / x._3)))


    threadGroupsWithWeights.toArray.sortBy(x => x._4).take(10).sortBy(x => -x._4).foreach(x => println("|" + x._1 + " | " + x._2.length + "  | " + x._3 + " | " + x._4 + " | "))
    //threadGroupsWithWeights.foreach(x => println(x._1 + "  time diff: " + x._3 + "  frequencies: " + x._2.length + "  weight: " + x._4))

    val threadBarPlotData = mailsGroupedByThread.map(x => (x._1, x._2.length)).toArray.sortBy(x => x._2)
    val threadDescriptions = threadBarPlotData.map(x => x._1)
    val threadValues = threadBarPlotData.map(x => Math.log1p(x._2.toDouble ))

    val weightedThreadBarPlotData = threadGroupsWithWeights.toArray.sortBy(x => x._4)
    val weightedThreadDescriptions = weightedThreadBarPlotData.map(x => x._1)
    val weightedThreadValues = weightedThreadBarPlotData.map(x => x._4)




    val StopWords = getStopWords
    val mailTDM = new TDM()
    //Build up the Term-Document Matrix for ham emails
    trainingData.foreach(x => x._5.split(" ").filter(_.nonEmpty).foreach(y => mailTDM.addTermToRecord(y, x._6)))
    //Sort the ham by total frequency for ease
    mailTDM.SortByOccurrenceRate(trainingData.size)
    //Filter out all stop words
    mailTDM.records = mailTDM.records.filter(x => !StopWords.contains(x.term))


    val barPlot = BarPlot.plot("Amount of emails per subject on log scale", weightedThreadValues, weightedThreadDescriptions)
    //Rotate the email addresses by -80 degrees such that we can read them
    barPlot.getAxis(0).setRotation(-1.3962634)
    barPlot.setAxisLabel(0, "")
    barPlot.setAxisLabel(1, "Weighted amount of mails per subject ")
    peer.setContentPane(barPlot)

    bounds = new Rectangle(800, 600)

    //   subjects.foreach(x => println(x._3))
  }


  def getStopWords: List[String] = {
    val source = scala.io.Source.fromFile(new File("/Users/mikedewaard/MachineLearning/Example Data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    lines.toList
  }


  def getFilesFromDir(path: String): List[File] = {
    val d = new File(path)
    if (d.exists && d.isDirectory) {
      //Remove the mac os basic storage file, and alternatively for unix systems "cmds"
      d.listFiles.filter(_.isFile).toList.filter(x => !x.toString.contains(".DS_Store") && !x.toString.contains("cmds"))
    } else {
      List[File]()
    }
  }


  def getFullEmail(file: File): String = {
    //Note that the encoding of the example files is latin1, thus this should be passed to the from file method.
    val source = scala.io.Source.fromFile(file)("latin1")
    val fullEmail = source.getLines mkString "\n"
    source.close()

    fullEmail
  }


  def getSubjectFromEmail(email: String): String = {

    //Find the index of the end of the subject line
    val subjectIndex = email.indexOf("Subject:")
    val endOfSubjectIndex = email.substring(subjectIndex).indexOf('\n') + subjectIndex

    //Extract the subject: start of subject + 7 (length of Subject:) until the end of the line.
    val subject = email.substring(subjectIndex + 8, endOfSubjectIndex).trim.toLowerCase

    //Additionally, we check whether the email was a response and remove the 're: ' tag, to make grouping on topic easier:
    subject.replace("re: ", "")
  }

  def getMessageBodyFromEmail(email: String): String = {

    val firstLineBreak = email.indexOf("\n\n")
    //Return the message body filtered by only text from a-z and to lower case
    email.substring(firstLineBreak).replace("\n", " ").replaceAll("[^a-zA-Z ]", "").toLowerCase
  }


  def getSenderFromEmail(email: String): String = {
    //Find the index of the From: line
    val fromLineIndex = email.indexOf("From:")
    val endOfLine = email.substring(fromLineIndex).indexOf('\n') + fromLineIndex

    //Search for the <> tags in this line, as if they are there, the email address is contained inside these tags
    val mailAddressStartIndex = email.substring(fromLineIndex, endOfLine).indexOf('<') + fromLineIndex + 1
    val mailAddressEndIndex = email.substring(fromLineIndex, endOfLine).indexOf('>') + fromLineIndex

    if (mailAddressStartIndex > mailAddressEndIndex) {

      //The email address was not embedded in <> tags, extract the substring without extra spacing and to lower case
      var emailString = email.substring(fromLineIndex + 5, endOfLine).trim.toLowerCase

      //Remove a possible name embedded in () at the end of the line, for example in test@test.com (tester) the name would be removed here
      val additionalNameStartIndex = emailString.indexOf('(')
      if (additionalNameStartIndex == -1) {
        emailString.toLowerCase
      }
      else {
        emailString.substring(0, additionalNameStartIndex).trim.toLowerCase
      }
    }
    else {
      //Extract the email address from the tags. If these <> tags are there, there is no () with a name in the From: string in our data
      email.substring(mailAddressStartIndex, mailAddressEndIndex).trim.toLowerCase
    }
  }

  def getDateFromEmail(email: String): Date = {
    //Find the index of the Date: line in the complete email
    val dateLineIndex = email.indexOf("Date:")
    val endOfDateLine = email.substring(dateLineIndex).indexOf('\n') + dateLineIndex

    //All possible date patterns in the emails.
    val datePatterns = Array("EEE MMM dd HH:mm:ss yyyy", "EEE, dd MMM yyyy HH:mm", "dd MMM yyyy HH:mm:ss", "EEE MMM dd yyyy HH:mm")

    datePatterns.foreach { x =>
      //Try to directly return a date from the formatting.when it fails on a pattern it continues with the next one until one works
      Try(return new SimpleDateFormat(x).parse(email.substring(dateLineIndex + 5, endOfDateLine).trim.substring(0, x.length)))
    }
    //Finally, if all failed return null (this will not happen with our example data but without this return the code will not compile)
    null
  }
}

