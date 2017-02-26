package statistical;

import java.util.Collection;

/**The Score class contains the basic _precision,
 * _recall, _f1 infrastructure
 * 
 * @author ccervantes
 */
public class Score 
{
	protected double _precision;
	protected double _recall;
	protected double _f1;
    protected ScoreType type;
	
	/**Empty constructor for child classes
	 * that use the init workflow instead
	 */
	public Score()
    {
        _precision = -1;
        _recall = -1;
        _f1 = -1;
        type = ScoreType.NONE;
    }

    public Score(double precision, double recall)
    {
        _precision = precision;
        _recall = recall;
        setF1();
    }
	
	/**Basic constructor requires <b>predictedCount</b>,
	 * <b>trueCount</b>, and <b>correctCount</b> to produce
	 * _f1 numbers
	 * 
	 * @param predictedCount - The number of items predicted for 
	 * 						   a class
	 * @param trueCount		 - The number of items in the gold
	 * @param correctCount   - The number of items that were correctly 
	 * 						   predicted (appeared in predicted and gold)
	 */
	public Score(int predictedCount, int trueCount, int correctCount)
	{
		_precision = predictedCount <= 0 ? 0 :
			(double)correctCount / (double)predictedCount;
		_recall = trueCount <= 0 ? 0 :
			(double)correctCount / (double)trueCount;
        setF1();
	}

	/**Creates a combined score object by 
	 * averaging the given scores, <b>s</b>
	 * 
	 * @param s - The Scores to combine
	 */
	public Score(Score... s)
	{
		double totalPrecision = 0.0;
		double totalRecall = 0.0;
		for(Score score : s) {
			totalPrecision += score._precision;
			totalRecall += score._recall;
		}
		_precision = totalPrecision / s.length;
		_recall = totalRecall / s.length;
        setF1();
	}

    public Score(Collection<Score> scores)
    {
        double totalPrecision = 0.0;
        double totalRecall = 0.0;
        for(Score score : scores) {
            totalPrecision += score._precision;
            totalRecall += score._recall;
        }
        _precision = totalPrecision / scores.size();
        _recall = totalRecall / scores.size();
        setF1();
    }

	/**Initializes this Score object based on two
	 * subordinate Score objects, <b>s1</b> and
	 * <b>s2</b> (written for BLANC)
	 * 
	 * @param s1 - First Score object
	 * @param s2 - Second Score object
	 */
	public void init(Score s1, Score s2)
	{
		_precision = (s1._precision + s2._precision) / 2;
		_recall = (s1._recall + s2._recall) / 2;
		_f1 = (s1._f1 + s2._f1) / 2;
	}

    protected void setF1()
    {
        if(_recall == 0 || _precision == 0)
            _f1 = 0;
        else
            _f1 = (2 * _precision * _recall) / (_precision + _recall);
    }
	
	/**Getters*/
	public double getPrecision(){return _precision;}
	public double getRecall(){return _recall;}
	public double getF1(){return _f1;}
    public ScoreType getType(){return type;}
	
	/**Returns a score string in the format
	 * "P: PPP.PP% | R: RRR.RR% | F1: FFF.FF%"
	 * 
	 * @return - The formatted score
	 */
	public String toScoreString()
	{
		return String.format("P: %6.2f%% | R: %6.2f%% | F1: %6.2f%%",
							 _precision *100, _recall *100, _f1 *100);
	}

    /**Returns a score string in the format
     * PPP.P\% & RRR.R\% & FFF.F\%\\
     *
     * @return
     */
    public String toLatexString()
    {
        return String.format("%.2f\\%% & %.2f\\%% & %.2f\\%%",
                             _precision * 100.0, _recall * 100.0,
                             _f1 * 100.0);
    }

    public enum ScoreType
    {
        BLANC, BCUBED, NONE;
    }
}
