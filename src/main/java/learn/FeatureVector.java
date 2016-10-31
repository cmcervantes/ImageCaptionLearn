package learn;

import java.util.*;

public class FeatureVector
{
    private Map<Integer, Double> _featureValueDict;
    public double label;
    public String comments;

    public FeatureVector()
    {
        _featureValueDict = new HashMap<>();
        label = 0.0;
        comments = "";
    }

    public FeatureVector(FeatureVector fv)
    {
        _featureValueDict = new HashMap<>();
        for(Integer idx : fv._featureValueDict.keySet())
            _featureValueDict.put(idx, fv._featureValueDict.get(idx));
        label = fv.label;
        comments = fv.comments;
    }

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

    public void addFeature(int idx, double value)
    {
        _featureValueDict.put(idx, value);
    }

    public List<Integer> getFeatureIndices()
    {
        List<Integer> indexList = new ArrayList<>(_featureValueDict.keySet());
        Collections.sort(indexList);
        return indexList;
    }
    public Double getFeatureValue(int idx) {return _featureValueDict.get(idx);}

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
            sb.append(label);

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
