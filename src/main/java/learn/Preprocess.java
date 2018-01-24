package learn;

import core.DocumentLoader;
import core.Main;
import nlptools.IllinoisAnnotator;
import nlptools.WordnetUtil;
import org.apache.commons.lang.ArrayUtils;
import structures.*;
import utilities.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

import static core.Main.flickr30k_lexicon;
import static core.Main.mscoco_lexicon;

/**Preprocessing functions typically generate files
 * for use with ImageCaptionLearn_py
 *
 * @author ccervantes
 */
public class Preprocess
{

    /***** Neural Network Preprocessing Functions ****/

    /**Strips punctuation tokens from the captions in the given docSet,
     * returning a mapping of caption IDs and punctuation-stripped
     * token lists
     *
     * @param docSet
     * @return
     */
    private static Map<String, List<Token>> stripPunctFromCaptions(Collection<Document> docSet)
    {
        Map<String, List<Token>> normCaptions = new HashMap<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                List<Token> tokens = new ArrayList<>();;
                for(Token t : c.getTokenList())
                    if(StringUtil.hasAlphaNum(t.toString()))
                        tokens.add(t);
                if(!tokens.isEmpty())
                    normCaptions.put(c.getUniqueID(), tokens);
            }
        }
        return normCaptions;
    }


    /**Re-maps mention boundaries to their position in the normalized (that is,
     * punctuation-less) captions
     *
     * @param docSet
     * @param normCaptions
     * @return
     */
    private static Map<String, int[]> remapMentionBounds(Collection<Document> docSet,
                                                         Map<String, List<Token>> normCaptions)
    {
        //Get a mapping from caption IDs to a list of mentions
        Map<String, List<Mention>> mentionsByCaption = new HashMap<>();
        for(Document d : docSet)
            for(Caption c : d.getCaptionList())
                mentionsByCaption.put(c.getUniqueID(), c.getMentionList());

        //Remap the mentions to their new bounds in the punctuation-less captions
        Map<String, int[]> mentionIndices = new HashMap<>();
        for(String capID : normCaptions.keySet()){
            List<Token> tokens = normCaptions.get(capID);
            for(int i=0; i<tokens.size(); i++){
                Token t = tokens.get(i);

                //If this token belongs to a mention, determine
                //if this is the first or last token in that mention
                if(t.mentionIdx > -1){
                    Mention m = mentionsByCaption.get(capID).get(t.mentionIdx);
                    if(!mentionIndices.containsKey(m.getUniqueID()))
                        mentionIndices.put(m.getUniqueID(), new int[]{-1, -1});
                    if(i == 0 || tokens.get(i-1).mentionIdx != t.mentionIdx)
                        mentionIndices.get(m.getUniqueID())[0] = i;
                    if(i == tokens.size()-1 || tokens.get(i+1).mentionIdx != t.mentionIdx)
                        mentionIndices.get(m.getUniqueID())[1] = i;
                }
            }
        }
        return mentionIndices;
    }

    public static void export_gloveTrainText(Collection<Document> docSet, String outroot)
    {
        Map<String, List<Token>> noPunctCaps =  stripPunctFromCaptions(docSet);
        StringBuilder text = new StringBuilder();
        for(String capID : noPunctCaps.keySet())
            text.append(StringUtil.listToString(noPunctCaps.get(capID), " "));
        FileIO.writeFile(text.toString(), outroot, "txt", false);
    }

    /**Exports a caption file for use with our neural networks, where
     * each caption has all punctuation tokens removed in the fomrat
     *      [img_id]#[cap_idx]      [cap_less_punc]
     *
     * @param docSet    Collection of Documents for which files will be
     *                  generated
     * @param outroot   Location to which the files should be saved
     */
    public static void export_neuralCaptionFile(Collection<Document> docSet, String outroot)
    {
        Map<String, List<Token>> normCaptions = stripPunctFromCaptions(docSet);
        List<String> ll_normCaptions = new ArrayList<>();
        normCaptions.forEach((k,v) -> ll_normCaptions.add(k + "\t" + StringUtil.listToString(v, " ")));
        FileIO.writeFile(ll_normCaptions, outroot + "_captions", "txt", false);
    }

    /**Exports three files for use with relation prediction via our neural networks:
     * mention index files for intra/cross captions respectively, and a mention pair
     * label file for our pairwise labeling, formatted as
     *      [pair_id]   [m1_start],[m1_end],[m2_start],[m2_end]   [label]
     * and
     *      [m_i_id] [m_j_id] [pairwise_label[
     * respectively
     *
     * UPDATE: also produces an ij intra caption file, which only contains
     *         indices for pairs where the first mention precedes the second
     *
     * @param docSet    Collection of Documents for which files will be
     *                  generated
     * @param outroot   Location to which the files should be saved
     */
    public static void export_neuralRelationFiles(Collection<Document> docSet,
                                                  String outroot)
    {
        DoubleDict<Integer> labelDistro = new DoubleDict<>();

        //Strip punctuation tokens from our captions
        Map<String, List<Token>> normCaptions = stripPunctFromCaptions(docSet);

        //Remap mention boundaries in these new normalized captions
        Map<String, int[]> mentionIndices = remapMentionBounds(docSet, normCaptions);

        //For each mention, we now have a mapping of IDs to normalized caption indices;
        //now we want to pair up these mentions (with intra-caption and cross-caption
        //mentions being stored separately)
        List<String> ll_mentionPairIndices_intra = new ArrayList<>();
        List<String> ll_mentionPairIndices_intra_ij = new ArrayList<>();
        List<String> ll_mentionPairIndices_cross = new ArrayList<>();
        List<String> ll_mentionPairLabels = new ArrayList<>();
        for(Document d : docSet){
            List<Mention> mentions = d.getMentionList();

            //Get predicted pronominal coref attachments
            //if this is an unreviewed image
            Map<Mention, String> pronomCorefDict = new HashMap<>();
            if(!d.reviewed)
                for(Mention[] pair : d.getPronomCorefMentions())
                    pronomCorefDict.put(pair[0], pair[1].getChainID());

            //Create a mapping from chains c_sub to all chains
            //c_sup for which c_sub is a subset
            Map<String, Set<String>> subsetChainDict = new HashMap<>();
            for(Chain[] subsetPair : d.getSubsetChains()){
                String subID = subsetPair[0].getID();
                if(!subsetChainDict.containsKey(subID))
                    subsetChainDict.put(subID, new HashSet<>());
                subsetChainDict.get(subID).add(subsetPair[1].getID());
            }

            for(int i=0; i<mentions.size(); i++){
                Mention m_i = mentions.get(i);

                //Skip any mentions which -- for some reason -- have
                //no alphanumeric characters
                int[] indices_i = mentionIndices.get(m_i.getUniqueID());
                if(indices_i == null)
                    continue;

                for(int j=i+1; j<mentions.size(); j++){
                    Mention m_j = mentions.get(j);

                    int[] indices_j = mentionIndices.get(m_j.getUniqueID());
                    if(indices_j == null)
                        continue;

                    //Simply store the first and last indices of ij and ji
                    Integer[] indices_ij = {indices_i[0], indices_i[1],
                                            indices_j[0], indices_j[1]};
                    Integer[] indices_ji = {indices_j[0], indices_j[1],
                                            indices_i[0], indices_i[1]};

                    //Find the label for this pair
                    int label_ij = 0, label_ji = 0;
                    String id_ij = Document.getMentionPairStr(m_i, m_j);
                    String id_ji = Document.getMentionPairStr(m_j, m_i);
                    String chainID_i = m_i.getChainID();
                    if(pronomCorefDict.containsKey(m_i))
                        chainID_i = pronomCorefDict.get(m_i);
                    String chainID_j = m_j.getChainID();
                    if(pronomCorefDict.containsKey(m_j))
                        chainID_j = pronomCorefDict.get(m_j);
                    Set<String> supChains_i = subsetChainDict.get(chainID_i);
                    Set<String> supChains_j = subsetChainDict.get(chainID_j);
                    if(!chainID_i.equals("0") && !chainID_j.equals("0")){
                        if(chainID_i.equals(chainID_j)){
                            label_ij = 1;
                            label_ji = 1;
                        } else if(supChains_i != null && supChains_i.contains(chainID_j)){
                            label_ij = 2;
                            label_ji = 3;
                        } else if(supChains_j != null && supChains_j.contains(chainID_i)){
                            label_ji = 2;
                            label_ij = 3;
                        }
                    }
                    labelDistro.increment(label_ij);
                    labelDistro.increment(label_ji);

                    //Get the pairwise label, for the label file
                    boolean nonvisMention = m_i.getChainID().equals("0") || m_j.getChainID().equals("0");
                    String gold = "null";
                    if(!nonvisMention){
                        if(label_ij == 1)
                            gold = "coref";
                        else if(label_ij == 2)
                            gold = "subsect_ij";
                        else if(label_ij == 3)
                            gold = "subset_ji";
                    }
                    ll_mentionPairLabels.add(String.format("%s %s %s", id_ij, id_ji, gold));

                    //Finally, add the lines to the appropriate line list
                    String line_ij = id_ij + "\t" + StringUtil.listToString(indices_ij, ",") +
                            "\t" + label_ij;
                    String line_ji = id_ji + "\t" + StringUtil.listToString(indices_ji, ",") +
                            "\t" + label_ji;
                    if(m_i.getCaptionIdx() == m_j.getCaptionIdx()){
                        ll_mentionPairIndices_intra.add(line_ij);
                        ll_mentionPairIndices_intra.add(line_ji);

                        //Because of the way we're iterating, we know that
                        //i comes before j, so we can always simply add ij pairs
                        //to the ij intra list
                        ll_mentionPairIndices_intra_ij.add(line_ij);
                    } else {
                        ll_mentionPairIndices_cross.add(line_ij);
                        ll_mentionPairIndices_cross.add(line_ji);
                    }
                }
            }
        }

        //Write all of the files
        FileIO.writeFile(ll_mentionPairLabels, outroot + "_mentionPair_labels", "txt", false);
        FileIO.writeFile(ll_mentionPairIndices_intra, outroot + "_mentionPairs_intra", "txt", false);
        FileIO.writeFile(ll_mentionPairIndices_intra_ij, outroot + "_mentionPairs_intra_ij", "txt", false);
        FileIO.writeFile(ll_mentionPairIndices_cross, outroot + "_mentionPairs_cross", "txt", false);

        Logger.log("Label Distribution");
        for(int i=0; i<labelDistro.keySet().size(); i++)
            System.out.printf("%d: %.2f%%\n",
                    i, 100.0 * labelDistro.get(i) / labelDistro.getSum());
    }

    /**Exports a mention file mapping the mention's ID to the normalized
     * caption bounds with nonvisual labeling; note also that predicted
     * coreferent pronouns are marked as nonvisual in the unreviewed
     * flickr training data; each line is formatted as
     *      [m_id]   [m_start],[m_end]   [label]
     *
     * @param docSet    Collection of Documents for which files will be
     *                  generated
     * @param outroot   Location to which the files should be saved
     */
    public static void export_neuralNonvisFile(Collection<Document> docSet, String outroot)
    {
        DoubleDict<Integer> labelDistro = new DoubleDict<>();

        //Strip punctuation tokens from our captions
        Map<String, List<Token>> normCaptions = stripPunctFromCaptions(docSet);

        //Remap mention boundaries in these new normalized captions
        Map<String, int[]> mentionIndices = remapMentionBounds(docSet, normCaptions);

        //For each mention, we now have a mapping of IDs to normalized caption indices;
        //now we want to pair up these mentions
        List<String> ll_mentionIndices = new ArrayList<>();
        for(Document d : docSet){

            //Get the attached pronominal mentions for unreviewed images
            Set<Mention> pronomCorefMentions = new HashSet<>();
            if(!d.reviewed)
                for(Mention[] pair : d.getPronomCorefMentions())
                    pronomCorefMentions.add(pair[0]);

            for(Mention m : d.getMentionList()){
                int nonvis = m.getChainID().equals("0") && !pronomCorefMentions.contains(m) ? 1 : 0;
                labelDistro.increment(nonvis);

                //We simply ignore those mentions that don't contain non-punct tokens
                if(mentionIndices.containsKey(m.getUniqueID())){
                    int[] indices = mentionIndices.get(m.getUniqueID());
                    ll_mentionIndices.add(String.format("%s\t%d,%d\t%s",
                            m.getUniqueID(), indices[0], indices[1], nonvis));
                }
            }
        }

        //Write all of the files
        FileIO.writeFile(ll_mentionIndices, outroot + "_mentions_nonvis", "txt", false);

        Logger.log("Label Distribution");
        for(int i=0; i<2; i++)
            System.out.printf("%d: %.2f%%\n",
                    i, 100.0 * labelDistro.get(i) / labelDistro.getSum());
    }


    /**Exports a mention file mapping the mention's ID to the normalized
     * caption bounds with cardinality labeling; each line is formatted as
     *      [m_id]   [m_start],[m_end]   [label]
     *
     * @param docSet    Collection of Documents for which files will be
     *                  generated
     * @param outroot   Location to which the files should be saved
     */
    public static void export_neuralCardinalityFile(Collection<Document> docSet, String outroot)
    {
        DoubleDict<Integer> labelDistro = new DoubleDict<>();

        //Strip punctuation tokens from our captions
        Map<String, List<Token>> normCaptions = stripPunctFromCaptions(docSet);

        //Remap mention boundaries in these new normalized captions
        Map<String, int[]> mentionIndices = remapMentionBounds(docSet, normCaptions);

        //Associate each mention's new boundaries with its cardinality label
        List<String> ll_mentionIndices = new ArrayList<>();
        for(Document d : docSet){
            //Get predicted pronominal coref attachments
            //if this is an unreviewed image
            Map<Mention, String> pronomCorefDict = new HashMap<>();
            if(!d.reviewed)
                for(Mention[] pair : d.getPronomCorefMentions())
                    pronomCorefDict.put(pair[0], pair[1].getChainID());

            //Map chain IDs to box counts
            Map<String, Integer> chainBoxCounts = new HashMap<>();
            for(Chain c : d.getChainSet())
                chainBoxCounts.put(c.getID(), Math.min(c.getBoundingBoxSet().size(), 11));

            for(Mention m : d.getMentionList()){
                String chainID = m.getChainID();
                if(pronomCorefDict.containsKey(m))
                    chainID = pronomCorefDict.get(m);

                int label = 0;
                if(!chainID.equals("0"))
                    label = chainBoxCounts.get(chainID);
                labelDistro.increment(label);

                //We simply ignore those mentions that don't contain non-punct tokens
                if(mentionIndices.containsKey(m.getUniqueID())){
                    int[] indices = mentionIndices.get(m.getUniqueID());
                    ll_mentionIndices.add(String.format("%s\t%d,%d\t%s",
                            m.getUniqueID(), indices[0], indices[1], label));
                }
            }
        }

        //Write all of the files
        FileIO.writeFile(ll_mentionIndices, outroot + "_mentions_card", "txt", false);

        Logger.log("Label Distribution");
        for(int i=0; i<labelDistro.size(); i++)
            System.out.printf("%d: %.2f%%\n",
                    i, 100.0 * labelDistro.get(i) / labelDistro.getSum());
    }

    /**Exports a file associating mention / box pairs with their affinity label
     *
     * @param docSet
     * @param outroot
     */
    public static void export_neuralAffinityFiles(Collection<Document> docSet, String outroot)
    {
        Logger.log("Initializing lexicons");
        Mention.initializeLexicons(Main.flickr30k_lexicon, Main.mscoco_lexicon);
        List<String> cocoCats = new ArrayList<>(Mention.getCOCOCategories());
        Collections.sort(cocoCats);

        Logger.log("Retrieving box categories");
        List<String> ll_boxCats = new ArrayList<>();
        for(Document d : docSet){
            for(BoundingBox b : d.getBoundingBoxSet()){
                int idx = cocoCats.indexOf(b.getCategory());
                Double[] onehot = new Double[cocoCats.size()];
                Arrays.fill(onehot, 0.0);
                if(idx >= 0)
                    onehot[idx] = 1.0;
                ll_boxCats.add(b.getUniqueID() + "\t" + StringUtil.listToString(onehot, ","));
            }
        }
        FileIO.writeFile(ll_boxCats, outroot + "_box_cats", "txt", false);

        Logger.log("Retrieving mention/box affinity labels");
        List<String> ll_affinityLabels = new ArrayList<>();
        DoubleDict<Integer> boxLabelHist = new DoubleDict<>();
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                for(BoundingBox b : d.getBoundingBoxSet()){
                    int label = !assocBoxes.isEmpty() && assocBoxes.contains(b) ? 1 : 0;
                    boxLabelHist.increment(label);
                    ll_affinityLabels.add(m.getUniqueID() + "|" + b.getUniqueID() +
                            "\t" + String.valueOf(label));
                }
            }
        }
        FileIO.writeFile(ll_affinityLabels, outroot + "_affinity_labels", "txt", false);

        Logger.log("Label Distribution");
        for(int i=0; i<boxLabelHist.size(); i++)
            System.out.printf("%d: %.2f%%\n",
                    i, 100.0 * boxLabelHist.get(i) / boxLabelHist.getSum());
    }

    /**Exports a file associating document IDs with the MPE label
     *
     * @param docSet
     * @param outroot
     */
    public static void export_neuralMPEFile(Collection<Document> docSet, String outroot)
    {
        String[] labels = {"contradiction", "neutral", "entailment"};
        List<String> ll_mpeLines = new ArrayList<>();
        DoubleDict<String> labelDistro = new DoubleDict<>();
        for(Document d : docSet){
            Map<String, String> commentDict = StringUtil.keyValStrToDict(d.comments);
            String label = commentDict.get("label");
            ll_mpeLines.add(d.getID() + "\t" + ArrayUtils.indexOf(labels, label));
            labelDistro.increment(label);
            labelDistro.increment("total");
        }
        FileIO.writeFile(ll_mpeLines, outroot + "_idx", "txt", false);

        Logger.log("Label Distribution");
        labelDistro.printPercentageDict("total");
    }

    /***** Phrase Localization Functions ****/

    /**Exports a box coordinate and image ID file for use with
     * the phrase localization FasterRCNN feature extraction
     *
     * @param docSet    Collection of Docuents (containing boxes)
     * @param outRoot   Root path to which '_boxCoords.csv' and
     *                  '_imageIDs.csv' will be appended
     */
    public static void export_phraseLocalization_boxFiles(Collection<Document> docSet,
                                                         String outRoot)
    {
        //Because we're putting these into matlab structures,
        //we have to be very mindful of order, so
        //iterate through documents in a reproducible way
        List<String> imageIDs = new ArrayList<>();
        for(Document d : docSet)
            imageIDs.add(d.getID());
        Collections.sort(imageIDs);

        //Store the documents in a dictionary (for the aforementioned
        //in-order retrieval)
        Map<String, Document> docDict = new HashMap<>();
        docSet.forEach(d -> docDict.put(d.getID(), d));

        //Retrieve box coordinates in document-then-index order
        List<String> coordList = new ArrayList<>();
        for(String imgID : imageIDs){
            Document d  = docDict.get(imgID);
            BoundingBox[] boxArr = new BoundingBox[d.getBoundingBoxSet().size()];
            for(BoundingBox b : d.getBoundingBoxSet())
                boxArr[b.getIdx()] = b;
            for(BoundingBox b : boxArr){
                Object[] params = {d.getID(), b.getIdx(), b.getXMin(),
                        b.getYMin(), b.getXMax(), b.getYMax()};
                coordList.add(StringUtil.listToString(params, ","));
            }
        }

        //save the files
        FileIO.writeFile(imageIDs, outRoot + "_imageIDs", "csv", false);
        FileIO.writeFile(coordList, outRoot + "_boxCoords", "csv", false);
    }

    /**Converts the boxFeatureFile (VGG box file: ~10G) to individual
     * document-level .feats formatted files (which makes the ccaLists
     * function much more memory efficient, if less CPU efficient)
     *
     * @param docSet         Document set for these bounding boxes
     * @param boxFeatureFile VGG bounding box feature file (from FasterRCNN part of
     *                       the phrase localization pipeline
     * @param boxFeatureDir  Directory in which to save all the document-level
     *                       box features
     */
    public static void export_phraseLocalization_convertBoxFeats(Collection<Document> docSet,
                                                                 String boxFeatureFile, String boxFeatureDir)
    {
        Logger.log("Reading box feats into memory");
        Map<String, int[]> boxIndexDict = new HashMap<>();
        Map<String, double[]> boxValueDict = new HashMap<>();
        try{
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(boxFeatureFile)));
            String nextLine = br.readLine();
            while(nextLine != null){
                String[] lineArr = nextLine.split(",");

                List<Integer> indices = new ArrayList<>();
                List<Double> values = new ArrayList<>();
                boolean invalidBox = false;
                for(int i=1; i<4097; i++){
                    if(lineArr[i].equals("%f")){
                        invalidBox = true;
                    } else {
                        double val = Double.parseDouble(lineArr[i]);
                        if(val != 0){
                            indices.add(i);
                            values.add(val);
                        }
                    }
                }

                if(invalidBox)
                    System.out.println("MALFORMED BOX FEATS: " + lineArr[0]);

                //Store this box's indices / values
                int[] indexArr = indices.stream().mapToInt(i->i).toArray();
                double[] valueArr = values.stream().mapToDouble(v->v).toArray();
                boxIndexDict.put(lineArr[0], indexArr);
                boxValueDict.put(lineArr[0], valueArr);

                //read the next line
                nextLine = br.readLine();
            }
        } catch(Exception ex){
            Logger.log(ex);
        }

        Logger.log("Processing box feats");
        int docIdx = 0;
        for (Document d : docSet){
            Set<FeatureVector> fvSet = new HashSet<>();
            for(BoundingBox b : d.getBoundingBoxSet()) {
                if(boxIndexDict.containsKey(b.getUniqueID()) &&
                   boxValueDict.containsKey(b.getUniqueID())){
                    int[] indices = boxIndexDict.get(b.getUniqueID());
                    double[] values = boxValueDict.get(b.getUniqueID());
                    fvSet.add(new FeatureVector(indices, values, 0, b.getUniqueID()));
                } else {
                    System.out.println("MISSING BOX FEATS: " + b.getUniqueID());
                }
            }

            List<String> ll_fv = new ArrayList<>();
            fvSet.forEach(fv -> ll_fv.add(fv.toString()));
            FileIO.writeFile(ll_fv, boxFeatureDir + d.getID().replace(".jpg", ""), "feats", false);
            docIdx++;
            Logger.logStatus("%.2f%% complete", 100.0 * (double)docIdx / docSet.size());
        }
    }

    /**Exports the phrase localization CCA lists, which vary based on data split;
     * training data is subsamples (so high-frequency phrases don't overwhelm
     * the model) while every mention/box pair is included in the dev/test
     * data so we can make predictions
     *
     * @param docSet            Documents from which to generate lists
     * @param dataSplit         Which data split this docSet contains
     *                          (train/dev/test)
     * @param boxFeatureDir     Directory containing bounding box features
     *                          (one image's boxes per file)
     * @param outRoot           Root path to which the lists will be written
     *                          (in the form &lt;outRoot&gt;_box_&lt;dataSplit&gt;.csv
     *                          and similar)
     */
    public static void export_phraseLocalization_ccaLists(Collection<Document> docSet, String dataSplit,
                                                          String boxFeatureDir, String outRoot)
    {
        //Initialize the lexicons
        Mention.initializeLexicons(flickr30k_lexicon, mscoco_lexicon);

        //Read bounding box filenames from the box feature dir to double check
        //the files will be there when we expect them
        Set<String> boxFiles = new HashSet<>();
        File boxDir = new File(boxFeatureDir);
        for(File f : boxDir.listFiles())
            if(f.isFile())
                boxFiles.add(f.getName().replace(".feats", ""));

        //Read stop words
        Set<String> stopWords =
                new HashSet<>(FileIO.readFile_lineList(Main.flickr30kResources +
                        "stop_words.txt"));

        //Store a mapping of [docID -> [mention -> [bounding boxes] ] ]
        Map<String, Map<Mention, Set<BoundingBox>>>
                mentionBoxesDict = new HashMap<>();
        for(Document d : docSet){
            if(boxFiles.contains(d.getID().replace(".jpg", ""))){
                for(Mention m : d.getMentionList()){
                    //Get this mention's string, removing stop words
                    StringBuilder normBuilder = new StringBuilder();
                    for(Token t : m.getTokenList()){
                        String normTok = t.toString().toLowerCase();
                        if(!stopWords.contains(normTok)){
                            normBuilder.append(normTok);
                            normBuilder.append(" ");
                        }
                    }
                    String normText = normBuilder.toString().trim();

                    if(!mentionBoxesDict.containsKey(normText))
                        mentionBoxesDict.put(normText, new HashMap<>());
                    mentionBoxesDict.get(normText).put(m, d.getBoxSetForMention(m));
                }
            } else {
                System.out.println("ERROR: found no box feats for " + d.getID());
            }
        }

        //store the in-order lists of box feats and text strings
        List<String> ll_img = new ArrayList<>(), ll_txt = new ArrayList<>();
        List<String> ll_ids = new ArrayList<>(), ll_labels = new ArrayList<>();
        List<String> ll_types_30k = new ArrayList<>(), ll_types_coco = new ArrayList<>();
        List<String> ll_types_coco_super = new ArrayList<>();
        DoubleDict<Integer> labelDistro = new DoubleDict<>();
        if(dataSplit.equals("train")){
            //Randomly sample 10 bounding boxes from those that have boxes
            Map<String, Set<BoundingBox>> mentionBoxesDict_sample =
                    new HashMap<>();
            for(String normText : mentionBoxesDict.keySet()){
                List<BoundingBox> boxList = new ArrayList<>();
                for(Mention m : mentionBoxesDict.get(normText).keySet())
                    boxList.addAll(mentionBoxesDict.get(normText).get(m));
                Collections.shuffle(boxList);
                if(!boxList.isEmpty()) {
                    mentionBoxesDict_sample.put(normText,
                            new HashSet<>(boxList.subList(0,
                                    Math.min(boxList.size() - 1, 10))));
                }
            }

            //store the in-order lists of box feats and text feats
            int text_idx = 0;
            for(String normText : mentionBoxesDict_sample.keySet()) {
                for(BoundingBox b : mentionBoxesDict_sample.get(normText)){
                    ll_txt.add(normText);

                    //Since we don't know which image a word's box will come from
                    //a-priori, open the doc's file at each box (less efficient, but necessary)
                    List<Double> imgFeats = null;
                    try{
                        BufferedReader br = new BufferedReader(new InputStreamReader(
                                new FileInputStream(boxFeatureDir +
                                        b.getDocID().replace(".jpg", ".feats"))));
                        String nextLine = br.readLine();
                        while(nextLine != null && imgFeats == null){
                            String fvID = nextLine.split(" # ")[1];
                            if(fvID.equals(b.getUniqueID())){
                                FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                                imgFeats = new ArrayList<>();
                                for(int i=1; i<=4096; i++)
                                    imgFeats.add(fv.getFeatureValue(i));
                            }
                            nextLine = br.readLine();
                        }
                        br.close();
                    } catch(Exception ex) {Logger.log(ex);}

                    //add the features to the img file
                    ll_img.add(StringUtil.listToString(imgFeats, ","));
                }
                text_idx++;
                Logger.logStatus("Completed %.2f%%", 100.0 * text_idx / mentionBoxesDict_sample.size());
            }
        } else {
            int docIdx = 0;
            for(Document d : docSet){
                //read box features for this file
                Map<String, List<Double>> boxFeatDict = new HashMap<>();
                List<String> ll_boxes = FileIO.readFile_lineList(boxFeatureDir +
                        d.getID().replace(".jpg", ".feats"));
                for(String featStr : ll_boxes){
                    FeatureVector fv = FeatureVector.parseFeatureVector(featStr);
                    List<Double> boxFeats = new ArrayList<>();
                    for(int i=1; i<=4096; i++)
                        boxFeats.add(fv.getFeatureValue(i));
                    boxFeatDict.put(fv.comments, boxFeats);
                }

                //for each mention / box pair
                for(Mention m : d.getMentionList()){
                    //Get this mention's string, removing stop words
                    StringBuilder normBuilder = new StringBuilder();
                    for(Token t : m.getTokenList()){
                        String normTok = t.toString().toLowerCase();
                        if(!stopWords.contains(normTok)){
                            normBuilder.append(normTok);
                            normBuilder.append(" ");
                        }
                    }
                    String normText = normBuilder.toString().trim();

                    //Add the types to the lists
                    String type_30k = m.getLexicalType();
                    if(type_30k.equals("other"))
                        type_30k = Mention.getLexicalEntry_flickr(m);
                    ll_types_30k.add(m.getUniqueID() + "," + type_30k);
                    String type_coco = Mention.getLexicalEntry_cocoCategory(m);
                    ll_types_coco.add(m.getUniqueID() + "," + type_coco);
                    String type_coco_super = Mention.getSuperCategory(type_coco);
                    ll_types_coco_super.add(m.getUniqueID() + "," + type_coco_super);

                            //Iterate through the box pairings
                    Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                    for(BoundingBox b : d.getBoundingBoxSet()){
                        ll_img.add(StringUtil.listToString(boxFeatDict.get(b.getUniqueID()), ","));
                        ll_txt.add(normText);
                        ll_ids.add(m.getUniqueID() + "|" + b.getUniqueID());
                        int label = assocBoxes.contains(b) ? 1 : 0;
                        ll_labels.add(String.valueOf(label));
                        labelDistro.increment(label);
                    }
                }
                docIdx++;
                Logger.logStatus("Completed %.2f%%", 100.0 * docIdx / docSet.size());
            }
        }

        //Save the files (train, being subsampled, has no id/label/type files
        FileIO.writeFile(ll_img, outRoot + "_box", "csv", false);
        FileIO.writeFile(ll_txt, outRoot + "_phrase", "txt", false);
        if(!dataSplit.equals("train")) {
            FileIO.writeFile(ll_ids, outRoot + "_id", "txt", false);
            FileIO.writeFile(ll_labels, outRoot + "_label", "txt", false);
            FileIO.writeFile(ll_types_30k, outRoot + "_type_30k", "csv", false);
            FileIO.writeFile(ll_types_coco, outRoot + "_type_coco", "csv", false);
            FileIO.writeFile(ll_types_coco_super, outRoot + "_type_coco_supercat", "csv", false);
        }

        //As a sanity check, print the label distribution
        Logger.log("Done; Label Distro:");
        System.out.print(labelDistro.toString());
    }

    /**Exports phrase localization CCA lists for full MSCOCO training, which assumes
     * every same-category box/mention are ground together (if no true grounding
     * is available) and subsamples the mention/box pairs
     *
     * @param docSet            Documents from which to generate lists
     * @param boxFeatureDir     Directory containing bounding box features
     *                          (one image's boxes per file)
     * @param outRoot           Root path to which the lists will be written
     *                          (in the form &lt;outRoot&gt;_box_&lt;dataSplit&gt;.csv
     *                          and similar)
     */
    public static void export_phraseLocalization_ccaLists(Collection<Document> docSet,
                                                          String boxFeatureDir, String outRoot)
    {
        //Initialize the lexicons
        Mention.initializeLexicons(flickr30k_lexicon, mscoco_lexicon);

        //Read bounding box filenames from the box feature dir to double check
        //the files will be there when we expect them
        Set<String> boxFiles = new HashSet<>();
        File boxDir = new File(boxFeatureDir);
        for(File f : boxDir.listFiles())
            if(f.isFile())
                boxFiles.add(f.getName().replace(".feats", ""));

        //Read stop words
        Set<String> stopWords =
                new HashSet<>(FileIO.readFile_lineList(Main.flickr30kResources +
                        "stop_words.txt"));

        //Store a mapping of [docID -> [mention -> [bounding boxes] ] ]
        Map<String, Map<Mention, Set<BoundingBox>>>
                mentionBoxesDict = new HashMap<>();
        for(Document d : docSet){
            if(boxFiles.contains(d.getID().replace(".jpg", ""))){
                for(Mention m : d.getMentionList()){
                    //Get this mention's string, removing stop words
                    StringBuilder normBuilder = new StringBuilder();
                    for(Token t : m.getTokenList()){
                        String normTok = t.toString().toLowerCase();
                        if(!stopWords.contains(normTok)){
                            normBuilder.append(normTok);
                            normBuilder.append(" ");
                        }
                    }
                    String normText = normBuilder.toString().trim();

                    if(!mentionBoxesDict.containsKey(normText))
                        mentionBoxesDict.put(normText, new HashMap<>());

                    //If this is a reviewed image, grab the mention's boxes; otherwise
                    //assume every same-category box is grounded to it
                    Set<BoundingBox> assocBoxes = new HashSet<>();
                    if(d.reviewed) {
                        assocBoxes.addAll(d.getBoxSetForMention(m));
                    } else {
                        String mentionCats = Mention.getLexicalEntry_cocoCategory(m);
                        if(mentionCats != null)
                            for(BoundingBox b : d.getBoundingBoxSet())
                                if(mentionCats.contains(b.getCategory()))
                                    assocBoxes.add(b);
                    }
                    mentionBoxesDict.get(normText).put(m, assocBoxes);
                }
            } else {
                System.out.println("ERROR: found no box feats for " + d.getID());
            }
        }

        //store the in-order lists of box feats and text strings
        List<String> ll_img = new ArrayList<>(), ll_txt = new ArrayList<>();
        DoubleDict<Integer> labelDistro = new DoubleDict<>();
        //Randomly sample 10 bounding boxes from those that have boxes
        Map<String, Set<BoundingBox>> mentionBoxesDict_sample =
                new HashMap<>();
        for(String normText : mentionBoxesDict.keySet()){
            List<BoundingBox> boxList = new ArrayList<>();
            for(Mention m : mentionBoxesDict.get(normText).keySet())
                boxList.addAll(mentionBoxesDict.get(normText).get(m));
            Collections.shuffle(boxList);
            if(!boxList.isEmpty()) {
                mentionBoxesDict_sample.put(normText,
                        new HashSet<>(boxList.subList(0,
                                Math.min(boxList.size() - 1, 10))));
            }
        }

        //store the in-order lists of box feats and text feats
        int text_idx = 0;
        for(String normText : mentionBoxesDict_sample.keySet()) {
            for(BoundingBox b : mentionBoxesDict_sample.get(normText)){
                ll_txt.add(normText);

                //Since we don't know which image a word's box will come from
                //a-priori, open the doc's file at each box (less efficient, but necessary)
                List<Double> imgFeats = null;
                try{
                    BufferedReader br = new BufferedReader(new InputStreamReader(
                            new FileInputStream(boxFeatureDir +
                                    b.getDocID().replace(".jpg", ".feats"))));
                    String nextLine = br.readLine();
                    while(nextLine != null && imgFeats == null){
                        String fvID = nextLine.split(" # ")[1];
                        if(fvID.equals(b.getUniqueID())){
                            FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                            imgFeats = new ArrayList<>();
                            for(int i=1; i<=4096; i++)
                                imgFeats.add(fv.getFeatureValue(i));
                        }
                        nextLine = br.readLine();
                    }
                    br.close();
                } catch(Exception ex) {Logger.log(ex);}

                //add the features to the img file
                ll_img.add(StringUtil.listToString(imgFeats, ","));
            }
            text_idx++;
            Logger.logStatus("Completed %.2f%%", 100.0 * text_idx / mentionBoxesDict_sample.size());
        }

        //Save the files (train, being subsampled, has no id/label/type files
        FileIO.writeFile(ll_img, outRoot + "_box", "csv", false);
        FileIO.writeFile(ll_txt, outRoot + "_phrase", "txt", false);

        //As a sanity check, print the label distribution
        Logger.log("Done; Label Distro:");
        System.out.print(labelDistro.toString());
    }

    /**Exports mention, box coordinate, and unique ID files for the given
     * document collection to the given out root; formerly called
     * bryan preproc, this function is intended for use with document collections
     * that we're passing to bryan to run his neural phrase image similarity
     * models on
     *
     */
    public static void export_phraseLocalization_affinityLists(Collection<Document> docSet,
                                                               String dataSplit, String outroot)
    {
        //store the in-order lists of box feats and text strings
        List<String> ll_img = new ArrayList<>(), ll_txt = new ArrayList<>();
        List<String> ll_ids = new ArrayList<>(), ll_labels = new ArrayList<>();
        DoubleDict<Integer> labelDistro = new DoubleDict<>();

        List<Document> docList = new ArrayList<>(docSet);
        for(int dIdx=0; dIdx < docList.size(); dIdx++) {
            Document d = docList.get(dIdx);

            for (Mention m : d.getMentionList()) {
                //Get this mention's string, keeping stop words for now
                String normText = m.toString().toLowerCase();

                Set<BoundingBox> boxSet = d.getBoundingBoxSet();
                Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                if(dataSplit.equals("train"))
                    boxSet = assocBoxes;

                for(BoundingBox b : boxSet) {
                    ll_txt.add(normText);
                    ll_img.add(String.format("%s %d %d %d %d", b.getDocID(),
                            b.getXMin(), b.getYMin(), b.getXMax(), b.getYMax()));

                    if(!dataSplit.equals("train")){
                        ll_ids.add(m.getUniqueID() + "|" + b.getUniqueID());
                        int label = assocBoxes.contains(b) ? 1 : 0;
                        ll_labels.add(String.valueOf(label));
                        labelDistro.increment(label);
                    }
                }
            }
            Logger.logStatus("Completed %.2f%%", 100.0 * dIdx / docSet.size());
        }

        //Save the files (train has no id /label files)
        FileIO.writeFile(ll_img, outroot + "_box", "csv", false);
        FileIO.writeFile(ll_txt, outroot + "_phrase", "txt", false);
        if(!dataSplit.equals("train")) {
            FileIO.writeFile(ll_ids, outroot + "_id", "txt", false);
            FileIO.writeFile(ll_labels, outroot + "_label", "txt", false);
            //As a sanity check, print the label distribution
            Logger.log("Done; Label Distro:");
            System.out.print(labelDistro.toString());
        }
    }

    /***** Histogram Functions (for later use as one-hot vectors) ****/

    /**Generates preposition histograms (for single mentions and mention pairs)
     * for use as one-hot feature vectors
     *
     * @param docSet    Collections of Documents on which to base the histograms
     *                  (typically training data only)
     */
    public static void export_prepositions(Collection<Document> docSet)
    {
        DoubleDict<String> prepHist = new DoubleDict<>();
        DoubleDict<String> prepPairHist = new DoubleDict<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                for(int i=0; i<c.getMentionList().size(); i++){
                    List<Chunk> chunkList_i = c.getMentionList().get(i).getChunkList();
                    Chunk left_i = null, right_i = null;
                    if(!chunkList_i.isEmpty()){
                        left_i = c.getLeftNeighbor(chunkList_i.get(0));
                        right_i = c.getRightNeighbor(chunkList_i.get(chunkList_i.size()-1));
                    }

                    if(left_i != null && left_i.getChunkType().equals("PP"))
                        prepHist.increment(left_i.toString().toLowerCase());
                    if(right_i != null && right_i.getChunkType().equals("PP"))
                        prepHist.increment(right_i.toString().toLowerCase());

                    for(int j=i+1; j<c.getMentionList().size(); j++){
                        List<Chunk> chunkList_j = c.getMentionList().get(j).getChunkList();
                        Chunk left_j = null, right_j = null;
                        if(!chunkList_j.isEmpty()){
                            left_j = c.getLeftNeighbor(chunkList_j.get(0));
                            right_j = c.getRightNeighbor(chunkList_j.get(chunkList_j.size()-1));
                        }

                        if(left_i != null && left_i.getChunkType().equals("PP") &&
                                left_j != null && left_j.getChunkType().equals("PP")){
                            String left_ij = left_i.toString() + "|" + left_j.toString();
                            String left_ji = left_j.toString() + "|" + left_i.toString();
                            prepPairHist.increment(left_ij.toLowerCase());
                            prepPairHist.increment(left_ji.toLowerCase());
                        }

                        if(right_i != null && right_i.getChunkType().equals("PP") &&
                                right_j != null && right_j.getChunkType().equals("PP")){
                            String right_ij = right_i.toString() + "|" + right_j.toString();
                            String right_ji = right_j.toString() + "|" + right_i.toString();
                            prepPairHist.increment(right_ij.toLowerCase());
                            prepPairHist.increment(right_ji.toLowerCase());
                        }
                    }
                }
            }
        }
        FileIO.writeFile(prepHist, "hist_preposition", "csv", false);
        FileIO.writeFile(prepPairHist, "hist_prepPair", "csv", false);
    }

    /**Generated modifier histograms (for numeric/non-numeric and
     * single/pairs of mentions) for use as one-hot feature fectors
     *
     * @param docSet    Collections of Documents on which to base the histograms
     *                  (typically training data only)
     */
    public static void export_modifiers(Collection<Document> docSet)
    {
        DoubleDict<String> modHist_numeric = new DoubleDict<>();
        DoubleDict<String> modHist_other = new DoubleDict<>();
        DoubleDict<String> modPairHist_numeric = new DoubleDict<>();
        DoubleDict<String> modPairHist_other = new DoubleDict<>();
        for(Document d : docSet){
            List<Mention> mentions = d.getMentionList();
            for(int i=0; i<mentions.size(); i++){
                Mention m1 = mentions.get(i);
                String[] mods_1 = m1.getModifiers();
                String numericMod_1 = mods_1[0];
                String mod_1 = mods_1[1];

                if(!numericMod_1.isEmpty())
                    modHist_numeric.increment(numericMod_1);
                if(!mod_1.isEmpty())
                    modHist_other.increment(mod_1);

                for(int j=i+1; j<mentions.size(); j++){
                    Mention m2 = mentions.get(j);
                    String[] mods_2 = m2.getModifiers();
                    String numericMod_2 = mods_2[0];
                    String mod_2 = mods_2[1];

                    if(!numericMod_2.isEmpty())
                        modHist_numeric.increment(numericMod_2);
                    if(!mod_2.isEmpty())
                        modHist_other.increment(mod_2);

                    if(!numericMod_1.isEmpty() || !numericMod_2.isEmpty()) {
                        modPairHist_numeric.increment(
                                StringUtil.getAlphabetizedPair(numericMod_1,
                                        numericMod_2));
                    }
                    if(!mod_1.isEmpty() || !mod_2.isEmpty()){
                        modPairHist_other.increment(mod_1 + "|" + mod_2);
                        modPairHist_other.increment(mod_2 + "|" + mod_1);
                    }
                }
            }
        }

        Logger.log("Writing files");
        FileIO.writeFile(modHist_numeric, "hist_numericModifier", "csv", false);
        FileIO.writeFile(modPairHist_numeric, "hist_numericModifierPair", "csv", false);
        FileIO.writeFile(modHist_other, "hist_modifier", "csv", false);
        FileIO.writeFile(modPairHist_other, "hist_modifierPair", "csv", false);
    }

    /**Generates hypernym histograms (for single mentions and mention pairs)
     * for use as one-hot feature vectors
     *
     * @param docSet    Collections of Documents on which to base the histograms
     *                  (typically training data only)
     */
    public static void export_hypernyms(Collection<Document> docSet)
    {
        WordnetUtil wnUtil = new WordnetUtil(Main.wordnetDir);
        DoubleDict<String> hypDict = new DoubleDict<>();
        DoubleDict<String> hypPairDict = new DoubleDict<>();
        Map<String, Set<String>> lemmaHypDict = new HashMap<>();
        for(Document d : docSet) {
            List<Mention> mentions = d.getMentionList();

            for (int i = 0; i < mentions.size(); i++) {
                Mention m_i = mentions.get(i);
                String lem_i = m_i.getHead().getLemma().toLowerCase();
                if (!lemmaHypDict.containsKey(lem_i)) {
                    Set<String> boh = wnUtil.getBagOfHypernyms(lem_i);
                    if (!boh.isEmpty())
                        lemmaHypDict.put(lem_i, boh);
                }

                Set<String> boh_i = new HashSet<>();
                if(lemmaHypDict.containsKey(lem_i))
                    boh_i = lemmaHypDict.get(lem_i);

                for (String h : boh_i)
                    hypDict.increment(h);

                for (int j = i + 1; j < mentions.size(); j++) {
                    Mention m_j = mentions.get(j);
                    String lem_j = m_j.getHead().getLemma().toLowerCase();

                    if (!lemmaHypDict.containsKey(lem_j)) {
                        Set<String> boh = wnUtil.getBagOfHypernyms(lem_j);
                        if (!boh.isEmpty())
                            lemmaHypDict.put(lem_j, boh);
                    }

                    Set<String> boh_j = new HashSet<>();
                    if(lemmaHypDict.containsKey(lem_j))
                        boh_j = lemmaHypDict.get(lem_j);

                    if (boh_i.isEmpty())
                        boh_i.add("");
                    if (boh_j.isEmpty())
                        boh_j.add("");
                    for (String h_i : boh_i)
                        for (String h_j : boh_j)
                            hypPairDict.increment(StringUtil.getAlphabetizedPair(h_i, h_j));
                }
            }
        }
        FileIO.writeFile(hypDict, "hist_hypernym", "csv", false);
        FileIO.writeFile(hypPairDict, "hist_hypernymPair", "csv", false);
    }

    /**Generates nonvisual histogram for use as one-hot feature vector
     *
     * @param docSet    Collections of Documents on which to base the histogram
     *                  (typically training data only)
     */
    public static void export_nonvisuals(Collection<Document> docSet)
    {
        DoubleDict<String> nonvisHist = new DoubleDict<>();
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                if(m.getChainID().equals("0") && m.getPronounType() == Mention.PRONOUN_TYPE.NONE){
                    String lemma = m.getHead().getLemma().toLowerCase();
                    if(StringUtil.hasAlphaNum(lemma))
                        nonvisHist.increment(lemma);
                }
            }
        }
        FileIO.writeFile(nonvisHist, "hist_nonvisual", "csv", false);
    }

    /**Exports coco category histogram files (to generate one-hots, etc.)
     *
     * @param docSet
     */
    public static void export_categories(Collection<Document> docSet)
    {
        Mention.initializeLexicons(flickr30k_lexicon, mscoco_lexicon);
        DoubleDict<String> catHist = new DoubleDict<>();
        DoubleDict<String> catPairHist = new DoubleDict<>();
        for(Document d : docSet){
            List<Mention> mentions = d.getMentionList();
            for(int i=0; i<mentions.size(); i++){
                Mention m_i = mentions.get(i);
                String cat_i = Mention.getLexicalEntry_cocoCategory(m_i);
                if(cat_i == null)
                    cat_i = "null";
                catHist.increment(cat_i);

                for(int j=i+1; j<mentions.size(); j++){
                    Mention m_j = mentions.get(j);
                    String cat_j = Mention.getLexicalEntry_cocoCategory(m_j);
                    if(cat_j == null)
                        cat_j = "null";

                    String cat_ij = cat_i + "|" + cat_j;
                    String cat_ji = cat_j + "|" + cat_i;
                    catPairHist.increment(cat_ij);
                    catPairHist.increment(cat_ji);
                }
            }
        }
        FileIO.writeFile(catHist, "hist_cocoCategory", "csv", false);
        FileIO.writeFile(catPairHist, "hist_cocoCategoryPair", "csv", false);
    }


    /***** Flickr30kEntities Caption-to-Databse Functions ****/

    /**Builds a SQLite database from the union of the .coref file and release
     * directory captions and annotations
     *
     * @param corefFile     Denotation Graph Generation or manual annotation .coref file
     * @param releaseDir    RELEASE directory containing Flickr30kEntities-style dataset info
     * @param commentsFile  Meta file containing annotator image comments
     * @param crossvalFile  Meta file containing cross-validation flags for each image
     * @param reviewedFile  Meta file containing whether a given image was reviewed
     * @param dbName        SQLite database filename
     */
    public static void buildImageCaptionDB(String corefFile, String releaseDir,
                                           String commentsFile, String crossvalFile,
                                           String reviewedFile, String dbName)
    {
        Collection<Document> docSet = getJointDocumentSet(corefFile, releaseDir,
                commentsFile, crossvalFile, reviewedFile);

        Logger.log("Uploading everything to the DB");
        DBConnector conn = new DBConnector(dbName);
        try{
            DocumentLoader.populateDocumentDB(conn, docSet, 100000, 1);
        } catch(Exception ex){
            utilities.Logger.log(ex);
        }
    }

    /**Builds an MySQL database from the union of the .coref file and release directory captions
     * and annotations
     *
     * @param corefFile     Denotation Graph Generation or manual annotation .coref file
     * @param releaseDir    RELEASE directory containing Flickr30kEntities-style dataset info
     * @param commentsFile  Meta file containing annotator image comments
     * @param crossvalFile  Meta file containing cross-validation flags for each image
     * @param reviewedFile  Meta file containing whether a given image was reviewed
     * @param host          MySQL host address
     * @param user          MySQL user name
     * @param password      MySQL password
     * @param name          MySQL database name
     */
    public static void buildImageCaptionDB(String corefFile, String releaseDir, String commentsFile,
                                           String crossvalFile, String reviewedFile, String host,
                                           String user, String password, String name)
    {
        Collection<Document> docSet = getJointDocumentSet(corefFile, releaseDir,
                commentsFile, crossvalFile, reviewedFile);

        Logger.log("Uploading everything to the DB");
        DBConnector conn = new DBConnector(host, user, password, name);
        try{
            DocumentLoader.populateDocumentDB(conn, docSet, 100000, 1);
        } catch(Exception ex){
            utilities.Logger.log(ex);
        }
    }

    /**Given a .coref file (graph generation or annotation pipeline), a release directory
     * (Flickr30kEntities pipeline), and meta files, returns a merged collection of
     * documents, where language information is taken from graph generation and
     * bounding box information taken from Flickr30kEntities
     *
     * @param corefFile     Denotation Graph Generation or manual annotation .coref file
     * @param releaseDir    RELEASE directory containing Flickr30kEntities-style dataset info
     * @param commentsFile  Meta file containing annotator image comments
     * @param crossvalFile  Meta file containing cross-validation flags for each image
     * @param reviewedFile  Meta file containing whether a given image was reviewed
     * @return              Collection of Documents with merged internal information
     */
    private static Collection<Document> getJointDocumentSet(String corefFile, String releaseDir,
                                                            String commentsFile, String crossvalFile,
                                                            String reviewedFile)
    {
        Logger.log("Loading documents from coref file");
        Collection<Document> docSet_coref = DocumentLoader.getDocumentSet(
                corefFile,
                flickr30k_lexicon, Main.flickr30kResources);
        Map<String, Document> docDict_coref = new HashMap<>();
        docSet_coref.forEach(d -> docDict_coref.put(d.getID(), d));

        Logger.log("Loading documents from flickr30kEntities file");
        Collection<Document> docSet_flickr =
                DocumentLoader.getDocumentSet(releaseDir,
                        Main.flickr30kResources);
        Map<String, Document> docDict_flickr = new HashMap<>();
        docSet_flickr.forEach(d -> docDict_flickr.put(d.getID(), d));

        Logger.log("Merging documents");
        for(String docID : docDict_coref.keySet())
            docDict_coref.get(docID).loadBoxesFromDocument(docDict_flickr.get(docID));

        Logger.log("Adding metadata (annotation comments, cross val flag, reviewed flag, and url)");
        String[][] commentsTable = FileIO.readFile_table(commentsFile);
        for(String[] row : commentsTable)
            docDict_coref.get(row[0] + ".jpg").comments = row[1];
        String[][] crossValTable = FileIO.readFile_table(crossvalFile);
        for(String[] row : crossValTable)
            docDict_coref.get(row[0] + ".jpg").crossVal = Integer.parseInt(row[1]);
        Set<String> reviewedImgs = new HashSet<>(FileIO.readFile_lineList(reviewedFile));
        for(String imgID : reviewedImgs)
            docDict_coref.get(imgID + ".jpg").reviewed = true;
        String urlRoot = "http://shannon.cs.illinois.edu/DenotationGraph/graph/flickr30k-images/";
        for(String docID : docDict_coref.keySet())
            docDict_coref.get(docID).imgURL = urlRoot + docID;

        return docDict_coref.values();
    }



    /***** MSCOCO Caption-to-Databse Functions ****/

    /**Reads MSCOCO sentences, predicts POS tags / chunk boundaries, and returns
     * caption objects
     *
     * @param posDir        Part-of-speech directory for the Illinois Tagger
     * @param chunkDir      Chunk directory for the Illinois Chunker
     * @param capFile_raw   Caption file containing raw, untokenized captions
     * @return              Dictionary of document IDs to arrays of Captions
     */
    private static Map<String, Caption[]> parseCocoCaptions(String posDir, String chunkDir, String capFile_raw)
    {
        Logger.log("Reading COCO data from file");
        Map<String, List<String>> captionDict = new HashMap<>();
        List<String> lineList = FileIO.readFile_lineList(capFile_raw);
        int idx = 0;
        for(String line : lineList){
            String[] lineParts = line.split("\t");
            String[] idParts = lineParts[0].split("#");
            String imgID = idParts[0];
            if(idParts.length < 2)
                System.out.println(idx + "\t" + line);
            int capIdx = Integer.parseInt(idParts[1]);

            List<String> captionList = captionDict.get(imgID);
            if(captionList == null)
                captionList = new ArrayList<>();
            int insertionIdx = 0;
            for(String cap : captionList)
                if(Integer.parseInt(cap.split("\t")[0].split("#")[1]) < capIdx)
                    insertionIdx++;
            captionList.add(insertionIdx, lineParts[1]);
            captionDict.put(imgID, captionList);
            idx++;
        }

        Logger.log("Initializing type lexicons");
        Mention.initializeLexicons(flickr30k_lexicon, null);

        Logger.log("Predicting pos tags and chunks");
        IllinoisAnnotator annotator = IllinoisAnnotator.createChunker(posDir, chunkDir);
        Map<String, Caption[]> capDict = new HashMap<>();
        int totalCaps = captionDict.keySet().size() * 5;
        int capCount = 0;
        for(String docID : captionDict.keySet()){
            Caption[] caps = new Caption[captionDict.get(docID).size()];
            for(int i=0; i<captionDict.get(docID).size(); i++)
                caps[i] = annotator.predictCaption(docID, i, captionDict.get(docID).get(i));
            capDict.put(docID, caps);
            capCount++;
            Logger.logStatus("Parsing; %.2f%% complete", 100.0 * (double)capCount / (double)totalCaps);
        }
        return capDict;
    }

    /**Given a set of MSCOCO Captions, applies the XofY fixes -- that is,
     * splitting [XofY] mentions or merging [X] of [Y] mentions --
     * and returns the resulting set of captions
     *
     * @param captions Set of Captions for which to fix the XofY mentions
     * @return         Fixed set of Captions
     */
    private static Set<Caption> applyXofYFixes(Set<Caption> captions)
    {
        Map<String, String> xofyTypeDict = new HashMap<>();
        String[][] xofyTable =
                FileIO.readFile_table(Main.mscocoResources +
                        "hist_xofy_lemmas_coco_20170423.csv");
        for(int i=1; i<xofyTable.length; i++){
            String[] row = xofyTable[i];
            if(!row[0].isEmpty() && !row[1].isEmpty())
                xofyTypeDict.put(row[0].replace("\"", ""), row[1].replace("\"", ""));
        }

        Set<Caption> newCaptions = new HashSet<>();
        for(Caption c : captions){
            List<Token> tokens_orig = c.getTokenList();
            List<Token> tokens_new = new ArrayList<>();
            int chunkIdx = 0;
            int mentionIdx = 0;
            boolean prevWasMerge = false;
            boolean prevWasSplit = false;

            for(int tIdx=0; tIdx<tokens_orig.size(); tIdx++){
                Token t = tokens_orig.get(tIdx);

                //If this token is "of", determine if the previous
                //lemma indicates a split or merge case
                boolean merge = false, split = false;
                if(t.toString().equals("of") && tIdx > 0 &&
                        tokens_orig.get(tIdx-1).mentionIdx >= 0){
                    String prevLem = tokens_orig.get(tIdx-1).getLemma().toLowerCase();
                    if(xofyTypeDict.containsKey(prevLem)){
                        switch(xofyTypeDict.get(prevLem)){
                            case "container":
                            case "portion":
                            case "collective":
                            case "quantifier":
                            case "quant-whole": merge = true;
                                break;
                            case "representation":
                                //representation merges occur _only_ with
                                //mentions after the first non pronominal
                                //to avoid "A picture of a man..." or
                                //"This is an image of a man..."
                                for(int i=0; i<t.mentionIdx; i++)
                                    if(c.getMentionList().get(i).getPronounType() == Mention.PRONOUN_TYPE.NONE)
                                        merge = true;
                                break;
                            case "part-of":
                            case "relation-near":
                            case "nonvisual": split = true;
                                break;
                        }
                    }
                    //This should only really be a merge case
                    //if this 'of' is outside of a mention
                    merge &= t.mentionIdx < 0;
                    //And should only be a split case if
                    //this 'of' is inside a mention
                    split &= t.mentionIdx >= 0;
                }

                //We advance the mention index when we encounter
                //a new mention, which occurs when
                //  - we're not currently splitting or merging
                //    (since those indicate 'of')
                //  - this token has a mention index
                //  - this token's mention index differs from
                //    the previous
                //  -or-
                //  - the previous action was to split
                // Note also that we do not advance the mention
                // idx if the previous action was to merge
                if(!merge && !split && tIdx > 0 && t.mentionIdx >= 0 &&
                        (t.mentionIdx != tokens_orig.get(tIdx-1).mentionIdx ||
                                prevWasSplit) && !prevWasMerge){
                    mentionIdx++;
                }

                //We advance the chunk index when we encounter
                //a new chunk, which occurs when
                // - this token has a chunk index
                // - this token's chunk index differs from the previous
                // -or- this is a split or merge of (such 'of's should
                //      always be a new PP chunk)
                // -or- the previous was a split/merge and this should
                //      be a new chunk
                //NOTE: sometimes we have an internal 'of' that
                //      we want to break into its own chunk, and this
                //      handles such a case
                if(t.chunkIdx >= 0 && tIdx > 0 &&
                        t.chunkIdx != tokens_orig.get(tIdx-1).chunkIdx ||
                        split || merge || prevWasSplit || prevWasMerge){
                    chunkIdx++;
                }

                //Typically we want to assign all attributes as-is except
                //the mention idx; those we assign according to two cases
                //1) This is a merging 'of'; it previously had no mention idx,
                //   now it should
                //2) This is _not_ a split 'of' and it had a previous
                //   mention idx
                int tMentionIdx = -1;
                if(merge || (!split && t.mentionIdx >= 0))
                    tMentionIdx = mentionIdx;

                //Assign the chunk index according to whether this token
                //was part of a chunk in the first place, except in the
                //split and merge cases
                int tChunkIdx = -1; String tChunkType = null;
                if(split || merge){
                    tChunkIdx = chunkIdx;
                    tChunkType = "PP";
                } else if(t.chunkIdx >= 0){
                    tChunkIdx = chunkIdx;
                    tChunkType = t.chunkType;
                }

                //Store our previous actions
                prevWasMerge = merge;
                prevWasSplit = split;

                //Wrap eveyrthing up into a new token
                tokens_new.add(new Token(t.getDocID(), t.getCaptionIdx(), tIdx, t.toString(),
                        t.getLemma(), tChunkIdx, tMentionIdx, tChunkType, t.getPosTag(), t.chainID));
            }
            newCaptions.add(new Caption(c.getDocID(), c.getIdx(), tokens_new));
        }
        return newCaptions;
    }

    /**Used to correct an issue with the php/javascript
     * annotation pipeline, as the internal 'of' chunks
     * aren't handled properly
     *
     * @param captions The captions to review
     * @return         The set of captions with this XofY
     *                 issue corrected
     */
    public static Set<Caption> applyXofYFixes_annotation(Set<Caption> captions)
    {
        Set<Caption> newCaptions = new HashSet<>();
        for(Caption c : captions) {
            List<Token> tokens_orig = c.getTokenList();
            List<Token> tokens_new = new ArrayList<>();

            int chunkIdx = 0;
            boolean prevInternalOf = false;
            for(int tIdx=0; tIdx<tokens_orig.size(); tIdx++){
                Token t = tokens_orig.get(tIdx);

                boolean internalOf = t.toString().equals("of") &&
                        t.mentionIdx >= 0 && !t.chunkType.equals("PP");

                //We advance the chunk index when we encounter
                //a new chunk, which occurs when
                // - this token has a chunk index
                // - this token's chunk index differs from the previous
                // -or- This token is a mention's internal 'of' that
                //      has the same chunk index
                // -or- The previous token was a mention's internal 'of'
                if(t.chunkIdx >= 0 && tIdx > 0 &&
                        t.chunkIdx != tokens_orig.get(tIdx-1).chunkIdx ||
                        internalOf || prevInternalOf){
                    chunkIdx++;
                }
                prevInternalOf = internalOf;

                //Other than that, we keep the indices the same;
                //the mention indices wont change with this operation
                int tChunkIdx = -1;
                String tChunkType = null;
                if(t.chunkIdx >= 0){
                    tChunkIdx = chunkIdx;
                    tChunkType = t.chunkType;
                    if(internalOf)
                        tChunkType = "PP";
                }

                //Wrap eveyrthing up into a new token
                tokens_new.add(new Token(t.getDocID(), t.getCaptionIdx(), tIdx,
                        t.toString(), t.getLemma(), tChunkIdx, t.mentionIdx,
                        tChunkType, t.getPosTag(), t.chainID));
            }
            newCaptions.add(new Caption(c.getDocID(), c.getIdx(), tokens_new));
        }
        return newCaptions;
    }

    /**Generates a .coref file -- reading COCO captions and performing
     * part-of-speech tagging and chunking -- and populates a database
     * with the captions
     *
     * @param posDir        Part-of-speech directory for the Illinois Tagger
     * @param chunkDir      Chunk directory for the Illinois Chunker
     * @param capFile_raw   Caption file containing raw, untokenized captions
     * @param capFile_coref File in which to save .coref formatted captions (after tagging/chunking)
     * @param conn          DBConnector specifying the DB in which to store the captions
     */
    public static void importCocoData_fromRaw(String posDir, String chunkDir,
                                              String capFile_raw, String capFile_coref,
                                              DBConnector conn)
    {
        /*
        String posDir = Main.dataPath + "pos/";
        String chunkDir = Main.dataPath + "chunk/";
        String cocoData = Main.mscocoPath + "coco_caps.txt";
        String corefFile = Main.mscocoPath + "coco_caps.coref";
        */

        Logger.log("Parsing raw captions for .coref file");
        Map<String, Caption[]> captionDict = parseCocoCaptions(posDir, chunkDir, capFile_raw);

        Logger.log("Applying XofY fixes");
        Set<Caption> captionSet = new HashSet<>();
        for(Caption[] captions : captionDict.values())
            for(Caption c : captions)
                captionSet.add(c);
        Set<Caption> captionSet_mod = applyXofYFixes(captionSet);

        Logger.log("Writing captions to " + capFile_coref);
        List<String> ll_caps = new ArrayList<>();
        for(Caption c : captionSet_mod)
            ll_caps.add(c.toCorefString());
        FileIO.writeFile(ll_caps, capFile_coref.replace(".coref", ""), "coref", false);

        //actually build the databse
        importCocoData(capFile_coref, conn);
    }

    /**Imports MSCOCO data into a database from a .coref file produced
     * using the DenotationGraph pipeline, applying xofy fixes and writing
     * a new .coref file as an intermediate step
     *
     * @param corefFile DenotationGraph .coref file to load to database
     * @param conn      DBConnector specifying the DB in which to store the captions
     */
    public static void importCocoData_fromCoref(String corefFile, DBConnector conn)
    {
        Mention.initializeLexicons(flickr30k_lexicon, null);
        Caption.initLemmatizer();
        Cardinality.initCardLists(Main.flickr30kResources +
                "collectiveNouns.txt");

        Logger.log("Loading Denotation Graph Generation's MSCOCO captions");
        Set<Caption> captions = new HashSet<>();
        for(String line : FileIO.readFile_lineList(corefFile)){
            try {
                captions.add(Caption.fromCorefStr(line));
            } catch(Exception ex){
                Logger.log(ex);
            }
        }

        Logger.log("Applying XofY fixes");
        Set<Caption> captions_mod = applyXofYFixes(captions);

        Logger.log("Writing new captions to .coref file");
        List<String> ll_newCaps = new ArrayList<>();
        for(Caption c : captions_mod)
            ll_newCaps.add(c.toCorefString(true));
        FileIO.writeFile(ll_newCaps,
                Main.mscocoPath + "coco_caps", "coref", true);

        Logger.log("Importing into DB");
        importCocoData(Main.mscocoPath + "coco_caps_" +
                Util.getCurrentDateTime("yyyMMdd") + ".coref", conn);
    }

    /**Given a .coref file, populates a database with its captions
     *
     * @param corefFile Caption file in .coref formatting to load to the DB
     * @param conn      DBConnector specifying the DB in which to store the captions
     */
    public static void importCocoData(String corefFile, DBConnector conn)
    {
        /*
        DBConnector conn = new DBConnector(Main.mscocoPath +
                "COCO_" + Util.getCurrentDateTime("yyyyMMdd") + ".db");
        DBConnector conn = new DBConnector(Main.flickr30k_mysqlParams[0],
                Main.flickr30k_mysqlParams[1], Main.flickr30k_mysqlParams[2],
                "ccervan2_coco");
        */

        Logger.log("Constructing document objects from files");
        Collection<Document> docSet =
                DocumentLoader.getDocumentSet(corefFile,
                        Main.mscocoPath + "coco_bbox.csv", Main.mscocoPath + "coco_imgs.txt",
                        flickr30k_lexicon, Main.flickr30kResources);
        try{
            DocumentLoader.populateDocumentDB(conn, docSet, 100000, 1);
        } catch(Exception ex){
            utilities.Logger.log(ex);
        }
        Logger.log("MSCOCO written to " +
                conn.getDBType().toString().toLowerCase() + " database");
    }


    /***** MPE File-to-Database Functions ****/

    /**Reads MPE data files (coref file and three mpe label files in the given
     * dir) and imports that data -- as documents -- into the database specified
     * by the connector
     *
     * @param corefFile
     * @param mpeDir
     * @param conn
     */
    public static void importMPEData(String corefFile, String mpeDir, DBConnector conn)
    {
        Logger.log("Inititalizing Lexicons, lists, etc.");
        Caption.initLemmatizer();
        Mention.initializeLexicons(flickr30k_lexicon, mscoco_lexicon);
        Cardinality.initCardLists(Main.flickr30kResources + "collectiveNouns.txt");

        //Loading the captions from the cap file, which contains
        //IDs from the MPE dataset, rather than from Flickr30k;
        //we'll need to remap these afterward
        Logger.log("Loading captions");
        Map<String, List<Caption>> docCaptionDict = new HashMap<>();
        List<String> mpeCorefLines = FileIO.readFile_lineList(corefFile);
        try {
            for(String corefLine : mpeCorefLines){
                String[] lineArr = corefLine.split("\t");
                String[] idArr = lineArr[0].split("#");
                String docID = idArr[0];
                if(!docCaptionDict.containsKey(docID))
                    docCaptionDict.put(docID, new ArrayList<>());
                docCaptionDict.get(docID).add(Caption.fromCorefStr(lineArr[1], docID,
                        docCaptionDict.get(docID).size()));
            }
        } catch(Exception ex){
            Logger.log(ex);
        }

        Logger.log("Loading Flickr30k Entities v1 data");
        DBConnector conn_v1 = new DBConnector(Main.flickr30k_sqlite_v1);
        Map<String, Caption> captionDict_v1 = new HashMap<>();
        for(Document d : DocumentLoader.getDocumentSet(conn_v1, -1))
            for(Caption c : d.getCaptionList())
                captionDict_v1.put(c.getUniqueID(), c);

        Logger.log("Loading mpe labels");
        String[] splits = {"dev", "train", "test"};
        Map<String, Integer> docSplitDict = new HashMap<>();
        Map<String, String> docLabelDict = new HashMap<>();
        Map<String, String> docIdDict = new HashMap<>();
        Map<String, String> mpeToV1Dict = new HashMap<>();
        DoubleDict<String> labelDistro = new DoubleDict<>();
        for(String split : splits){
            List<String> mpeLabelLines = FileIO.readFile_lineList(mpeDir + "mpe_" + split + ".txt");

            //skip the first line, which is column headers
            for(int i=1; i<mpeLabelLines.size(); i++){
                String[] mpeLabelArr = mpeLabelLines.get(i).split("\t");

                //Since we know each mpe item is sourced from a single image
                //and images don't get reused, we can grab the doc ID from any
                //of the premises and map it to the MPE id
                String mpeID = mpeLabelArr[0];
                for(int j=0; j<4; j++)
                    mpeToV1Dict.put(mpeID + "#" + j, mpeLabelArr[j+1].split("/")[0]);
                docIdDict.put(mpeID, mpeLabelArr[1].split("#")[0]);

                //Get the label off the end of the line
                String label = mpeLabelArr[mpeLabelArr.length-1];
                labelDistro.increment(label);

                //Put the cross-val flag and the label in their dictionaries
                docSplitDict.put(mpeID, ArrayUtils.indexOf(splits, split));
                docLabelDict.put(mpeID, label);
            }
        }

        //Print the label distro as a sanity check
        double total = labelDistro.getSum();
        for(String label : labelDistro.keySet())
            System.out.printf("%s: %.2f%%\n", label, 100.0 * labelDistro.get(label) / total);

        Logger.log("Packaging everything up into Documents");
        Set<Document> docSet = new HashSet<>();
        for(String id : docCaptionDict.keySet()){
            if(!docSplitDict.containsKey(id)){
                System.out.println("Could not find " + id);
                for(String idStr : docSplitDict.keySet()){
                    System.out.println(idStr);
                    System.exit(0);
                }
            }

            // Re-construct the MPE captions with v1 chain IDs,
            // for the 97.53% of mentions for which we can find matches
            List<Caption> captions_mpe = docCaptionDict.get(id);
            for(Caption c_mpe : captions_mpe){
                // While all premise captions are accounted for in
                // the v1 data (I checked), the hypotheses aren't
                if(!mpeToV1Dict.containsKey(c_mpe.getUniqueID()))
                    continue;

                Caption c_v1 = captionDict_v1.get(mpeToV1Dict.get(c_mpe.getUniqueID()));
                List<Mention> mentions_mpe = c_mpe.getMentionList();
                List<Mention> mentions_v1 = c_v1.getMentionList();
                if(mentions_mpe.size() == mentions_v1.size()){
                    for(int i=0; i<mentions_mpe.size(); i++){
                        Mention m_mpe = mentions_mpe.get(i);
                        Mention m_v1 = mentions_v1.get(i);
                        if(m_mpe.toString().equals(m_v1.toString())){
                            m_mpe.setChainID(m_v1.getChainID());
                        }
                    }
                }
            }
            Document d  = new Document(id, docCaptionDict.get(id));
            d.crossVal = docSplitDict.get(id);
            d.comments = String.format("orig_id:%s;label:%s",
                    docIdDict.get(id), docLabelDict.get(id));
            docSet.add(d);
        }

        Logger.log("Populating database");
        try{
            DocumentLoader.populateDocumentDB(conn, docSet, 100000, 1);
        } catch(Exception ex){
            utilities.Logger.log(ex);
        }
        Logger.log("MPE written to " +
                conn.getDBType().toString().toLowerCase() + " database");
    }

}
