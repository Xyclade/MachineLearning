import scala.collection.mutable

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
  def SortByTotalFrequency() = records = records.sortBy( x => -x.totalFrequency)
  def SortByOccurrenceRate(rate : Int) = records = records.sortBy( x => -x.occurrenceRate(rate))
}

class TDMRecord(val term : String, var occurrences :  mutable.HashMap[String,Int] )
{
  def totalFrequency = occurrences.map(y => y._2).fold(0){ (z, i) => z + i}
  def occurrenceRate(totalDocuments : Int) : Double  = occurrences.size.toDouble / totalDocuments
  def densityRate(totalTerms : Int) : Double  = totalFrequency.toDouble / totalTerms
}