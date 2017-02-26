package learn;

import structures.Document;
import structures.Mention;
import utilities.DoubleDict;
import utilities.FileIO;

public class BinaryClassifierScoreDict extends DoubleDict<String>
{
    /**Loads the score dict from <b>scoreFile</b>, assuming
     * the format is in
     *      ID,score
     *      doc:[docID];[otherKeys]:[otherVals],[-1,1]
     *
     * @param scoreFile
     */
    public BinaryClassifierScoreDict(String scoreFile)
    {
        super();
        if(scoreFile != null && !scoreFile.isEmpty()){
            String[][] scoreTable = FileIO.readFile_table(scoreFile);
            for(String[] scoreRow : scoreTable)
                this.increment(scoreRow[0], Double.parseDouble(scoreRow[1]));
        }
    }

    /**Creates an empty score dict object into which scores will be added
     * and presumably exported to a file
     */
    public BinaryClassifierScoreDict()
    {
        super();
    }

    /**Sets the value of the given mention pair's score
     * to the given score
     *
     * @param m1    - Mention 1
     * @param m2    - Mention 2
     * @param score - Score between mentions 1 and 2
     */
    public void addScore(Mention m1, Mention m2, Double score)
    {
        set(Document.getMentionPairStr(m1, m2), score);
    }

    /**Increments the value of the given key's score
     * to the given score
     *
     * @param key   - Item key
     * @param score - Score of the item
     */
    public void addScore(String key, Double score)
    {
        set(key, score);
    }

    /**Returns the stored score for the given pair of mentions
     * (<b>m1</b>,<b>m2</b>).
     *
     * @param m1    - Mention 1
     * @param m2    - Mention 2
     * @return      - The normalized score (null if not found)
     */
    public Double get(Mention m1, Mention m2)
    {
        return dict.get(Document.getMentionPairStr(m1,m2));
    }

    /**Returns the stored score for the given mention
     *
     * @param m
     * @return
     */
    public Double get(Mention m){return dict.get(m.getUniqueID());}

    /**Returns the [-1,1] score for a given predicted label (<b>yhat</b>)
     * and probability array (<b>probabilities</b>); in practice this
     * function is merely a wrapper for
     *      -1*(1-yhat)*probabilities[yhat] + (yhat)probabilities[yhat]
     *
     * Added for liblinear integration
     *
     * @param yhat          - predicted label (idx into <b>probabilities</b>)
     * @param probabilities - probability arr
     * @return              - normalized [-1,1] score, based on label
     */
    public static Double getNormScore(int yhat, double[] probabilities)
    {
        return -1*(1-yhat)*probabilities[yhat] + yhat*probabilities[yhat];
    }
}
