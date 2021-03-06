package learn;

import java.util.*;

/**A holdover from when learning took place in Java,
 * FeatureVectors enable very sparse vectors of
 * features (think million-entry one-hots) in a
 * moderately memory-efficient way; the formatting
 * is borrowed from LibLinear's expected format,
 * which is
 *      label idx:val idx:val ... idx:val # comments
 *
 * @author ccervantes
 */
public class FeatureVector
{
    private Map<Integer, Double> _featureValueDict;
    public double label;
    public String comments;

    /**Default constructor
     *
     */
    public FeatureVector()
    {
        _featureValueDict = new HashMap<>();
        label = 0.0;
        comments = "";
    }

    /**Initializes this feature vector with
     * the values from another
     *
     * @param fv
     */
    public FeatureVector(FeatureVector fv)
    {
        _featureValueDict = new HashMap<>();
        for(Integer idx : fv._featureValueDict.keySet())
            _featureValueDict.put(idx, fv._featureValueDict.get(idx));
        label = fv.label;
        comments = fv.comments;
    }

    /**Creates a feature vector with a given dense list of
     * numbers
     *
     * @param denseVector
     * @param label
     * @param comments
     */
    public FeatureVector(List<Double> denseVector, double label, String comments)
    {
        _featureValueDict = new HashMap<>();
        for(int i=0; i<denseVector.size(); i++){
            //indices always start at 1 for this representation
            _featureValueDict.put(i+1, denseVector.get(i));
        }
        this.label = label;
        this.comments = comments;
    }

    /**Constructor directly using arrays of indices and values,
     * mimicking the behavior of calling addFeature repeatedly
     *
     * @param indices
     * @param values
     */
    public FeatureVector(int[] indices, double[] values, double label, String comments)
    {
        this.label = label;
        this.comments = comments;

        _featureValueDict = new HashMap<>();
        for(int i=0; i<indices.length; i++)
            addFeature(indices[i], values[i]);
    }

    /**Adds a feature with the given index and value
     *
     * @param idx
     * @param value
     */
    public void addFeature(int idx, double value)
    {
        _featureValueDict.put(idx, value);
    }

    /**Returns the feature indices that don't have 0 values
     *
     * @return
     */
    public List<Integer> getFeatureIndices()
    {
        List<Integer> indexList = new ArrayList<>(_featureValueDict.keySet());
        Collections.sort(indexList);
        return indexList;
    }

    /**Returns the feature value at the given index
     *
     * @param idx
     * @return
     */
    public Double getFeatureValue(int idx)
    {
        if(_featureValueDict.containsKey(idx))
            return _featureValueDict.get(idx);
        return 0.0;
    }

    /**Returns this feature vector's values as a dense vector
     *
     * @return
     */
    public List<Double> toDenseVector()
    {
        int maxIdx = Integer.MIN_VALUE;
        for(int idx : _featureValueDict.keySet())
            if(idx > maxIdx)
                maxIdx = idx;

        List<Double> fvList = new ArrayList<>();
        for(int i=1; i<=maxIdx; i++){
            double val = 0.0;
            if(_featureValueDict.containsKey(i))
                val = _featureValueDict.get(i);
            fvList.add(val);
        }
        return fvList;
    }

    /**Returns the string representation of this feature vector
     *      label idx:val idx:val ... idx:val # comments
     *
     * @return
     */
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();

        //add the label
        if(label == 1)
            sb.append("+1");
        else if(label == 0)
            sb.append("0");
        else if(label == -1)
            sb.append("-1");
        else
            sb.append((int)label);

        //and the featureIdx:value pairs
        for(Integer idx : _featureValueDict.keySet()){
            sb.append(" ");
            sb.append(idx);
            sb.append(":");
            sb.append(_featureValueDict.get(idx));
        }

        //and finally append any comments, if present
        if(!comments.isEmpty()) {
            sb.append(" # ");
            sb.append(comments);
        }

        return sb.toString();
    }

    /**Parses a feature vector from a given string
     *
     * @param s
     * @return
     */
    public static FeatureVector parseFeatureVector(String s)
    {
        FeatureVector fv = new FeatureVector();

        //split the line into comments / not comments
        String[] commentSplit = s.split(" # ");
        if(commentSplit.length>1)
            fv.comments = commentSplit[1].trim();

        //split the not-comments by spaces
        String[] vectorSplit = commentSplit[0].trim().split(" ");

        //set the vector's label
        fv.label = Double.parseDouble(vectorSplit[0]);

        //add each of the features to the vector
        for(int i=1; i<vectorSplit.length; i++){
            String[] featureSplit = vectorSplit[i].trim().split(":");
            int idx = Integer.parseInt(featureSplit[0]) + 1;
            double value = Double.parseDouble(featureSplit[1]);
            fv.addFeature(idx, value);
        }

        return fv;
    }
}
