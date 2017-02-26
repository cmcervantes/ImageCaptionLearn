package core;

import learn.BinaryClassifierScoreDict;
import learn.ClassifyUtil;
import learn.RelationInference;
import learn.WekaMulticlass;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.*;
import net.sourceforge.argparse4j.internal.HelpScreenException;
import net.sourceforge.argparse4j.internal.UnrecognizedArgumentException;
import org.apache.commons.lang.ArrayUtils;
import out.OutTable;
import statistical.ScoreDict;
import structures.BoundingBox;
import structures.Chain;
import structures.Document;
import structures.Mention;
import utilities.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;


/**The Overlord is responsible for
 * parsing args and calling other PanOpt
 * modules
 * 
 * @author ccervantes
 */
public class Overlord 
{
    public static String[] mysql_params = {"engr-cpanel-mysql.engr.illinois.edu", "ccervan2_root",
                                           "thenIdefyheaven!", "ccervan2_imageCaption"};
    public static String dataPath = "/home/ccervan2/source/data/";
    public static String lexPath = "/shared/projects/DenotationGraphGeneration/data/lexiconsNew/";
    public static String captionTePath = "/shared/projects/caption_te/";
    public static String wordnetDir = "/shared/data/WordNet-3.0/dict/";
    public static String word2vecPath = "/shared/projects/word2vec/word2vec.vector.gz";
    public static String datasetPath = "/shared/projects/Flickr30kEntities_v2/";
    public static String datasetPath_legacy = "/shared/projects/Flickr30kEntities/";
    public static String dbPath = datasetPath + "Flickr30kEntities_v2_20170209.db";
    public static String resourcesDir = datasetPath + "resources/";
    public static String dbPath_legacy = datasetPath_legacy + "Flickr30kEntities_v1.db";
    public static String boxFeatureDir = dataPath + "Flickr30kEntities_v1/box_feats/";
    public static String boxMentionDir = dataPath + "Flickr30kEntities_v2/box_mention_embeddings/";

	private static String _outroot;

    private static void _debug()
    {
        DBConnector conn = new DBConnector("engr-cpanel-mysql.engr.illinois.edu",
                "ccervan2_root", "thenIdefyheaven!", "ccervan2_imageCaption");
        Collection<Document> docSet = DocumentLoader.getDocumentSet(conn, 0, 100);
        BinaryClassifierScoreDict nonvisScores =
                new BinaryClassifierScoreDict("/Users/syphonnihil/source/working/flickr30kEntities_v2_nonvis_dev.scores");
        for(Document d : docSet){
            Set<String> pred_nonvis = new HashSet<>();
            Set<String> gold_nonvis = new HashSet<>();
            for(Mention m : d.getMentionList()){
                if(nonvisScores.get(m) != null && nonvisScores.get(m) > 0){
                    pred_nonvis.add(m.toString());
                }
                if(m.getChainID().equals("0")){
                    gold_nonvis.add(m.toString());
                }
            }

            if(!gold_nonvis.isEmpty()){
                System.out.println(d.getID());
                System.out.println("\t pred: " + StringUtil.listToString(pred_nonvis, "|"));
                System.out.println("\t gold: " + StringUtil.listToString(gold_nonvis, "|"));
            }
        }

        System.exit(0);
        for(Document d : docSet){
            Set<Mention> visualPronoms_gold = new HashSet<>();
            for(Mention m : d.getMentionList()){
                if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                   m.getPronounType() != Mention.PRONOUN_TYPE.SEMI &&
                   !m.getChainID().equals("0")){
                    visualPronoms_gold.add(m);
                }
            }

            Set<Mention> visualPronoms_pred = new HashSet<>();
            Set<String> predCorefPairs = ClassifyUtil.pronominalCoref(d, new HashSet<>(d.getMentionList()));
            List<Mention> mentions = d.getMentionList();
            for(int i=0; i<mentions.size(); i++) {
                Mention m_i = mentions.get(i);
                for (int j = i + 1; j < mentions.size(); j++) {
                    Mention m_j = mentions.get(j);
                    String id_ij = Document.getMentionPairStr(m_i, m_j);
                    String id_ji = Document.getMentionPairStr(m_j, m_i);

                    if(predCorefPairs.contains(id_ij) || predCorefPairs.contains(id_ji)){
                        if(m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                           m_i.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                            visualPronoms_pred.add(m_i);
                        if(m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                           m_j.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                            visualPronoms_pred.add(m_j);
                    }
                }
            }

            Set<Mention> intersect = new HashSet<>(visualPronoms_gold);
            intersect.retainAll(visualPronoms_pred);

            if(!visualPronoms_gold.isEmpty() && !visualPronoms_pred.isEmpty()){
                if(visualPronoms_gold.size() > intersect.size()){
                    System.out.println("Gold not pred");
                    for(Mention m : visualPronoms_gold)
                        if(!intersect.contains(m))
                            System.out.println(m.toString() + " | " + m.toDebugString() + " | " + m.getPronounType().toString());
                }

                if(visualPronoms_pred.size() > intersect.size()){
                    System.out.println("Pred not Gold");
                    for(Mention m : visualPronoms_pred)
                        if(!intersect.contains(m))
                            System.out.println(m.toString() + " | " + m.toDebugString() + " | " + m.getPronounType().toString());
                }
            }
        }

    }

	/**Main function. Parses <b>args</b> and calls
	 * other PanOpt modules
	 * 
	 * @param args - Command line arguments
	 */
	public static void main(String[] args)
	{
	    /*
        _debug();
	    System.exit(0);*/

        //start our runtime clock
		Logger.startClock();

		//arguments for the  manager
		ArgumentParser parser = 
				ArgumentParsers.newArgumentParser("ImageCaptionLearn");
		parser.defaultHelp(true);
		parser.description("ImageCaptionLearn has several " +
                "modules, each with their own args. To get " +
                "more info on a particular module, specify " +
                "the module with --help");
        _setArgument_flag(parser.addArgument("--quiet"),
                "Whether to log output");
        _setArgument(parser.addArgument("--log_delay"),
                "Minimum seconds to wait between logging progress status messages",
                Integer.class, 90);
        _setArgument(parser.addArgument("--out"),
                "Writes output to file with ROOT prefix", "ROOT");
        _setArgument(parser.addArgument("--threads"),
                "Uses NUM threads, where applicable",
                Integer.class, 1, "NUM", false);
        String[] dataOpts = {"train", "dev", "test", "all"};
        _setArgument_opts(parser.addArgument("--data"), dataOpts, null,
                "Where applicable, loads only the specified data");
        _setArgument(parser.addArgument("--rand_docs"),
                "Loads NUM random documents of the given data type",
                Integer.class, null, "NUM", false);
        _setArgument_flag(parser.addArgument("--local"), "Runs locally; queries mysqlDB");

        //Throwaway arguments will appear in the Debug module;
        //More permanant things will appear in their own modules
        Subparsers subparsers = parser.addSubparsers();
        Subparser debugParser = subparsers.addParser("Debug");
        _setArgument_flag(debugParser.addArgument("--heterog_subset"),
                "Exports a list of heterog subset pairs");
        _setArgument_flag(debugParser.addArgument("--penult_filter"),
                "Exports mention pairs for which the penult filter fires");
        _setArgument_flag(debugParser.addArgument("--mod_subset"),
                "Exports newly modified subset features");

        /* Data Group */
        Subparser dataParser = subparsers.addParser("Data");
        _setArgument(dataParser.addArgument("--convert_to_arff"),
                "Converts the specified .feats file to .arff format",
                "PATH");
        String[] featOpts = {"relation", "affinity", "nonvis", "box_card"};
        _setArgument_opts(dataParser.addArgument("--extractFeats"), featOpts, null,
                "Extracts features to --out");
        _setArgument_opts(dataParser.addArgument("--buildDB"), new String[]{"mysql", "sqlite"}, "mysql",
                "Rebuilds the specified database in the default locations");

        /* Learn Group */
        Subparser learnParser = subparsers.addParser("Learn");
        _setArgument(learnParser.addArgument("--train_file"), "Training data", "PATH");
        _setArgument(learnParser.addArgument("--model_file"), "Model file (save to / load from)", "PATH");
        String[] learners = {"weka_multi", "liblinear_logistic"};
        _setArgument_opts(learnParser.addArgument("--learner"), learners, "weka_multi",
                "Specifies which training is taking place");
        _setArgument(learnParser.addArgument("--batch_size"), "train arg; uses SIZE batches",
                Integer.class, 100, "SIZE", false);
        _setArgument(learnParser.addArgument("--epochs"), "train arg; run for NUM epochs",
                Integer.class, 1000, "NUM", false);
        _setArgument(learnParser.addArgument("--eval_file"), "Evaluation data", "PATH");
        _setArgument_flag(learnParser.addArgument("--pronom_coref"),
                "Evaluates rule-based pronominal coreference resolution");

        /* Infer Group */
        Subparser infParser = subparsers.addParser("Infer");
        _setArgument(infParser.addArgument("--nonvis_scores"),
                "Nonvisual scores file; associates mention unique IDs with [0,1] nonvis prediction",
                "FILE");
        _setArgument(infParser.addArgument("--relation_scores"),
                "Pairwise relation scores file; associates mention pair IDs with [0,1] (n,c,b,p) predictions",
                String.class, "/home/ccervan2/source/data/feats/flickr30kEntities_v2_relation_dev.scores", "FILE",
                false);
        _setArgument(infParser.addArgument("--affinity_scores"),
                "Affinity scores file; associates mention|box unique IDs with [0,1] affinity prediction",
                String.class, "/home/ccervan2/source/data/feats/flickr30kEntities_v2_affinity_dev.scores",
                "FILE", false);
        _setArgument(infParser.addArgument("--cardinality_scores"),
                "Cardinality scores file; associates mention unique IDs with [0,1] "+
                "(0-10,11+) cardinality prediction", String.class,
                "/home/ccervan2/source/data/feats/flickr30kEntities_v2_box_card_dev.scores",
                "FILE", false);
        _setArgument_opts(infParser.addArgument("--inf_type"), new String[]{"relation", "grounding", "combined"},
                "relation", "Specified which inference module to use");
        _setArgument_flag(infParser.addArgument("--type_constraint"), "Enables the inference type constraint");
        _setArgument_flag(infParser.addArgument("--export_files"), "Writes examples to out/coref/htm/ and conll "+
                "files to out/coref/conll/");
        _setArgument(infParser.addArgument("--graph_root"), "Loads previous inference "+
                "graphs from ROOT_relation.obj and ROOT_grounding.obj", "ROOT");
        _setArgument(infParser.addArgument("--alpha"), "[0,1] alpha value", Double.class, 0.0, "NUM", false);

        //Actually parse the arguments
        Namespace ns = _parseArgs(parser, args);

		//parse our main args
		if(!ns.getBoolean("quiet"))
			Logger.setVerbose();
        boolean runLocal = ns.getBoolean("local");
		_outroot = ns.getString("out");
        int numThreads = ns.getInt("threads");
        Logger.setStatusDelay(ns.getInt("log_delay"));
        String dataSplit = ns.getString("data");
        Collection<Document> docSet = null;
        if(dataSplit != null){
            DBConnector conn;
            if(runLocal){
                conn = new DBConnector(mysql_params[0], mysql_params[1], mysql_params[2],mysql_params[3]);
            } else {
                conn = new DBConnector(dbPath);
            }

            Integer numRandImgs = ns.getInt("rand_docs");
            Integer crossValFlag = null;
            switch(dataSplit){
                case "dev": crossValFlag = 0; break;
                case "train": crossValFlag = 1; break;
                case "test": crossValFlag = 2; break;
            }

            //get all documents
            if(crossValFlag == null && numRandImgs == null) {
                docSet = DocumentLoader.getDocumentSet(conn);
            } else if(crossValFlag != null){
                if(numRandImgs == null)
                    docSet = DocumentLoader.getDocumentSet(conn, crossValFlag);
                else
                    docSet = DocumentLoader.getDocumentSet(conn, crossValFlag, numRandImgs);
            }
        }

        //Switch on the specified module, and parse module args
        List<String> argList = Arrays.asList(args);
        if(argList.contains("Debug")){
            if(ns.getBoolean("heterog_subset")) {
                Minion.export_subsetHeterogType(docSet);
            } else if(ns.getBoolean("penult_filter")){
                Minion.export_penultFilter(docSet);
            } else if(ns.getBoolean("mod_subset")){
                Minion.export_modSubsetFeats(docSet, dataSplit);
            } else {

                BinaryClassifierScoreDict nonvisScores =
                        new BinaryClassifierScoreDict("/home/ccervan2/source/data/feats/nonvis_test_20170215.scores");
                ScoreDict<Integer> nonvisScoreDict_model = new ScoreDict<>();
                ScoreDict<Integer> nonvisScoreDict_heur = new ScoreDict<>();
                String[][] nonvisTable = FileIO.readFile_table(Overlord.resourcesDir + "hist_nonvisual.csv");
                Set<String> nonvisHeads = new HashSet<>();
                for(String[] row : nonvisTable){
                    if(Double.parseDouble(row[1]) > 10)
                        nonvisHeads.add(row[0]);
                }

                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        int gold = m.getChainID().equals("0") ? 1 : 0;
                        int pred_model = nonvisScores.get(m) >= 0 ? 1 : 0;
                        int pred_heur = nonvisHeads.contains(m.getHead().toString().toLowerCase()) ? 1 : 0;
                        nonvisScoreDict_model.increment(gold, pred_model);
                        nonvisScoreDict_heur.increment(gold, pred_heur);
                    }
                }
                System.out.println("NONVIS MODEL");
                nonvisScoreDict_model.printCompleteScores();
                System.out.println("NONVIS HEURISTIC");
                nonvisScoreDict_heur.printCompleteScores();
                System.exit(0);


                /*
                Set<String> interestingIDs = new HashSet<>();
                for(Document d : docSet){
                    boolean hasSubset = d.getSubsetChains().size() > 0;
                    int numSemiPronom = 0, numPronom = 0, numNonvis = 0, numMultibox = 0;
                    for(Mention m : d.getMentionList()){
                        if(m.getPronounType() == Mention.PRONOUN_TYPE.SEMI)
                            numSemiPronom++;
                        else if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                            numPronom++;

                        if(m.getChainID().equals("0"))
                            numNonvis++;

                        if(d.getBoxSetForMention(m).size() > 1)
                            numMultibox++;
                    }

                    if(hasSubset && numSemiPronom > 1 && numPronom > 1 && numNonvis > 1 && numMultibox > 1){
                        interestingIDs.add("\"" + d.getID() + "\"");
                    }
                }
                System.out.println("{" + StringUtil.listToString(interestingIDs, ", ") + "}");
                System.exit(0);
                */

                String intraScoreFile = Overlord.dataPath + "feats/relation_dev_20170220_intra.scores";
                Map<String, double[]> intraScores = ClassifyUtil.readMccScoresFile(intraScoreFile);
                System.out.println(intraScores.size());
                System.out.println(new ArrayList<>(intraScores.keySet()).get(0));
                OutTable ot_intra = new OutTable("doc_id", "cap_idx", "m_1_idx", "m_2_idx",
                        "m_1", "m_2", "gold", "pred", "n", "c", "b", "p", "caption");
                for(Document d : docSet){
                    List<Mention> mentions = d.getMentionList();
                    Set<String> subsetMentions = d.getSubsetMentions();
                    for(int i=0; i<mentions.size(); i++){
                        Mention m_i = mentions.get(i);
                        for(int j=i+1; j<mentions.size(); j++){
                            Mention m_j = mentions.get(j);

                            if(m_i.getCaptionIdx() != m_j.getCaptionIdx())
                                continue;

                            String id_ij = Document.getMentionPairStr(m_i, m_j);
                            String id_ji = Document.getMentionPairStr(m_j, m_i);

                            int gold_ij = -1, gold_ji = -1;
                            if(!m_i.getChainID().equals("0") && !m_j.getChainID().equals("0")){
                                gold_ij = 0; gold_ji = 0;
                                if(m_i.getChainID().equals(m_j.getChainID())){
                                    gold_ij = 1; gold_ji = 1;
                                } else if(subsetMentions.contains(id_ij)){
                                    gold_ij = 2; gold_ji = 3;
                                } else if(subsetMentions.contains(id_ji)){
                                    gold_ji = 2; gold_ij = 3;
                                }
                            }

                            double[] scores_ij = intraScores.get(id_ij);
                            double[] scores_ji = intraScores.get(id_ji);
                            if(scores_ij != null && scores_ji != null) {
                                if(gold_ij > 0 || Util.getMaxIdx(ArrayUtils.toObject(scores_ij)) > 0)
                                    ot_intra.addRow(d.getID(), m_i.getCaptionIdx(),
                                        m_i.getIdx(), m_j.getIdx(), m_i.toString(),
                                        m_j.toString(), gold_ij,
                                        Util.getMaxIdx(ArrayUtils.toObject(scores_ij)),
                                        scores_ij[0], scores_ij[1], scores_ij[2],
                                        scores_ij[3], d.getCaption(m_i.getCaptionIdx()));
                                if(gold_ji > 0 || Util.getMaxIdx(ArrayUtils.toObject(scores_ji)) > 0)
                                    ot_intra.addRow(d.getID(), m_j.getCaptionIdx(),
                                        m_j.getIdx(), m_i.getIdx(), m_j.toString(),
                                        m_i.toString(), gold_ji,
                                        Util.getMaxIdx(ArrayUtils.toObject(scores_ji)),
                                        scores_ji[0], scores_ji[1],
                                        scores_ji[2], scores_ji[3], d.getCaption(m_j.getCaptionIdx()));
                            }
                        }
                    }
                }
                ot_intra.writeToCsv("ex_intra_cap_rel", true);
                System.exit(0);










                String relationFile = "/home/ccervan2/source/data/feats/flickr30kEntities_v2_relation_dev.scores";
                String affinityFile = "/home/ccervan2/source/data/feats/flickr30kEntities_v2_affinity_dev.scores";
                String cardinalityFile = "/home/ccervan2/source/data/feats/flickr30kEntities_v2_box_card_dev.scores";
                String typeCostFile = Overlord.resourcesDir + "hist_typePairLogProb.csv";
                Map<String, double[]> affinityScoreDict = ClassifyUtil.readMccScoresFile(affinityFile);
                Map<String, double[]> cardScoreDict = ClassifyUtil.readMccScoresFile(cardinalityFile);

                for(Document d : docSet){
                    if(d.getID().equals("2934801096.jpg")){
                        for(String pairID : d.getSubsetMentions()){
                            Map<String, String> dict = StringUtil.keyValStrToDict(pairID);
                            System.out.println(dict.get("caption_1") + dict.get("mention_1") + "|" +
                                    dict.get("caption_2") + dict.get("mention_2"));
                        }

                        for(Mention m : d.getMentionList()){
                            if(cardScoreDict.containsKey(m.getUniqueID())){
                                System.out.print("" + m.getCaptionIdx() + m.getIdx() + "\t");
                                double[] scores = cardScoreDict.get(m.getUniqueID());
                                for(double score : scores)
                                    System.out.print(score + " | ");
                                System.out.println();
                            }
                            Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);

                            for(BoundingBox b : d.getBoundingBoxSet()){
                                String id = m.getUniqueID() + "|" + b.getUniqueID();
                                if(affinityScoreDict.containsKey(id)){
                                    System.out.print("" + m.getCaptionIdx() + m.getIdx() + "|" + b.getIdx() + "\t");
                                    System.out.print(assocBoxes.contains(b) + "\t");
                                    double[] scores = affinityScoreDict.get(id);
                                    for(double score : scores)
                                        System.out.print(score + " | ");
                                    System.out.println();
                                }
                            }
                        }
                    }
                }
                System.exit(0);


                String cardScoreFile = Overlord.dataPath + "feats/flickr30kEntities_v2_box_card_dev.scores";
                Map<String, double[]> cardScores = ClassifyUtil.readMccScoresFile(cardScoreFile);
                OutTable ot_card = new OutTable("doc_id", "cap_idx", "m_idx", "mention", "gold_card", "pred_card",
                        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11+");
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                        if(cardScores.containsKey(m.getUniqueID())){
                            double[] scores = cardScores.get(m.getUniqueID());
                            ot_card.addRow(d.getID(), m.getCaptionIdx(), m.getIdx(),
                                    m.toString(), assocBoxes.size(), Util.getMaxIdx(ArrayUtils.toObject(scores)),
                                    scores[0], scores[1], scores[2], scores[3], scores[4], scores[5],
                                    scores[6], scores[7], scores[8], scores[9], scores[10], scores[11]);
                        }
                    }
                }
                ot_card.writeToCsv("ex_card", true);






                List<String> docIds = new ArrayList<>();
                docIds.add("2934801096.jpg");
                DBConnector conn = new DBConnector(dbPath);
                docSet = DocumentLoader.getDocumentSet(conn, docIds);



                /*
                RelationInference relInf = new RelationInference(docSet, null,
                            relationFile, typeCostFile, null, null,
                            false, 0.0);
                relInf.infer(numThreads, false);
                relInf.evaluate(false);

                relInf = new RelationInference(docSet, null,
                        null, null, affinityFile, cardinalityFile,
                        false, 0.0);
                relInf.infer(numThreads, false);
                relInf.evaluate(false);
                */
                RelationInference relInf = new RelationInference(docSet, null,
                        relationFile, typeCostFile, affinityFile, cardinalityFile,
                        null, 0.0);
                relInf.infer(numThreads, false);
                relInf.evaluate(false);
                System.exit(0);



                //Minion.export_typePairFreq(docSet);







                DoubleDict<String> chainPairs = new DoubleDict<>();
                for(Document d : docSet){
                    Set<Chain[]> subsetChains = d.getSubsetChains();

                    List<Chain> chains = new ArrayList<>(d.getChainSet());
                    for(int i=0; i<chains.size(); i++){
                        Chain c_i = chains.get(i);
                        Set<BoundingBox> boxes_i = c_i.getBoundingBoxSet();

                        for(int j=i+1; j<chains.size(); j++){
                            Chain c_j = chains.get(j);
                            Set<BoundingBox> boxes_j = c_j.getBoundingBoxSet();

                            Set<BoundingBox> intersect = new HashSet<>(boxes_i);
                            intersect.retainAll(boxes_j);

                            if(intersect.size() == boxes_i.size() && intersect.size() == boxes_j.size()) {
                                if(Util.containsArr(subsetChains, new Chain[]{c_i, c_j})){
                                    chainPairs.increment("subset_sameBox");
                                } else if(Util.containsArr(subsetChains, new Chain[]{c_j, c_i})){
                                    chainPairs.increment("subset_sameBox");
                                } else {
                                    chainPairs.increment("sameBox");
                                }
                            } else {
                                chainPairs.increment("diffBoxes");
                            }
                        }
                    }
                }
                double totalBoxes = chainPairs.getSum();
                for(String key : chainPairs.keySet())
                    chainPairs.divide(key, totalBoxes);
                System.out.println(chainPairs);
                System.exit(0);


                DoubleDict<String> heterogTypeDict = new DoubleDict<>();
                double total = 0.0;
                for(Document d : docSet){
                    for(Chain c : d.getChainSet()){
                        Set<String> types = new HashSet<>();
                        for(Mention m : c.getMentionSet())
                            if(m.getPronounType() == Mention.PRONOUN_TYPE.NONE)
                                types.add(m.getLexicalType());
                        List<String> typeList = new ArrayList<>(types);
                        Collections.sort(typeList);
                        if(typeList.size() > 1){
                            if(typeList.size() != 2 || !typeList.contains("other"))
                                heterogTypeDict.increment(StringUtil.listToString(typeList, "|"));
                        }
                    }
                    total += d.getChainSet().size();
                }
                for(String types : heterogTypeDict.keySet())
                    heterogTypeDict.divide(types, total);
                FileIO.writeFile(heterogTypeDict, "hist_heterogChains_deleteme", "csv", true);
                System.out.println(heterogTypeDict.getSum());
                System.exit(0);





                DoubleDict<Integer> mentionPairs = new DoubleDict<>();
                DoubleDict<Integer> mentionBoxPairs = new DoubleDict<>();
                DoubleDict<Integer> combined = new DoubleDict<>();
                for(Document d : docSet){
                    int mentions = d.getMentionList().size();
                    int boxes = d.getBoundingBoxSet().size();
                    mentionPairs.increment(mentions * mentions);
                    mentionBoxPairs.increment(mentions * boxes);
                    combined.increment(mentions * mentions * boxes);
                }
                System.out.println("------- grounding ------");
                List<Integer> keys = new ArrayList<>(mentionBoxPairs.keySet());
                Collections.sort(keys);
                for(Integer key : keys)
                    System.out.println(key + ": " + mentionBoxPairs.get(key));
                System.out.println("------- relation ------");
                keys = new ArrayList<>(mentionPairs.keySet());
                Collections.sort(keys);
                for(Integer key : keys)
                    System.out.println(key + ": " + mentionPairs.get(key));
                System.out.println("------- combined ------");
                keys = new ArrayList<>(combined.keySet());
                Collections.sort(keys);
                for(Integer key : keys)
                    System.out.println(key + ": " + combined.get(key));


                System.exit(0);





                //Minion.export_nonvisuals(docSet);

                DoubleDict<String> heterogTypes = new DoubleDict<>();
                int totalChains = 0;
                for(Document d : docSet){
                    Set<String> types = new HashSet<>();
                    totalChains += d.getChainSet().size();
                    for(Chain c : d.getChainSet()){
                        for(Mention m : c.getMentionSet())
                            types.add(m.getLexicalType());
                        if(types.size() > 1){
                            List<String> typeList = new ArrayList<>(types);
                            Collections.sort(typeList);
                            heterogTypes.increment(StringUtil.listToString(typeList, "|"));
                        }
                    }
                }
                for(String t : heterogTypes.keySet())
                    heterogTypes.divide(t, totalChains);
                System.out.println(heterogTypes);
                System.out.println(heterogTypes.getSum());

                System.exit(0);


                BinaryClassifierScoreDict nonvis_scoreDict =
                        new BinaryClassifierScoreDict(Overlord.dataPath + "feats/flickr30kEntities_v2_nonvis_dev.scores");

                for(Document d : docSet){
                    Set<Mention> gold_nonvis = new HashSet<>();
                    Set<Mention> pred_nonvis = new HashSet<>();
                    for(Mention m : d.getMentionList()){
                        if(m.getChainID().equals("0"))
                            gold_nonvis.add(m);
                        if(nonvis_scoreDict.get(m) > 0)
                            pred_nonvis.add(m);
                    }
                    Set<Mention> intersect = new HashSet<>(gold_nonvis);
                    intersect.retainAll(pred_nonvis);

                    if(!intersect.isEmpty() && gold_nonvis.size() != pred_nonvis.size()){
                        System.out.println("---" + d.getID() + "----");
                        System.out.println("Gold Nonvis: " + StringUtil.listToString(gold_nonvis, " | "));
                        System.out.println("Pred Nonvis: " + StringUtil.listToString(pred_nonvis, " | "));
                    }
                }
                System.exit(0);
            }
        } else if(argList.contains("Data")) {
            String featsFileToConvert = ns.getString("convert_to_arff");
            String featsToExtract = ns.getString("extractFeats");
            String buildDB = ns.getString("buildDB");

            if(featsFileToConvert != null) {
                WekaMulticlass.exportToArff(featsFileToConvert);
            } else if(featsToExtract != null){
                if(featsToExtract.equals("relation"))
                    ClassifyUtil.exportFeatures_relation(docSet, _outroot, numThreads);
                else if(featsToExtract.equals("affinity"))
                    ClassifyUtil.exportFeatures_affinity(docSet, dataSplit);
                else if(featsToExtract.equals("nonvis")) {
                    ClassifyUtil.exportFeatures_nonvis(docSet, _outroot);
                } else if(featsToExtract.equals("box_card")) {
                    ClassifyUtil.exportFeatures_boxCard(docSet, _outroot);
                }
            } else if(buildDB != null){
                System.out.println("WARNING: there's a bug where certain cardinalities are null");
                if(buildDB.equals("mysql")){
                    Minion.buildImageCaptionDB(Overlord.datasetPath + "Flickr30kEntities_v2.coref",
                            Overlord.datasetPath + "RELEASE/", Overlord.resourcesDir + "img_comments.csv",
                            Overlord.resourcesDir + "img_crossval.csv", Overlord.resourcesDir + "img_reviewed.txt",
                            mysql_params[0], mysql_params[1], mysql_params[2], mysql_params[3]);
                } else if(buildDB.equals("sqlite")){
                    Minion.buildImageCaptionDB(Overlord.datasetPath + "Flickr30kEntities_v2.coref",
                            Overlord.datasetPath + "RELEASE/", Overlord.resourcesDir + "img_comments.csv",
                            Overlord.resourcesDir + "img_crossval.csv", Overlord.resourcesDir + "img_reviewed.txt",
                            Overlord.datasetPath + "Flickr30kEntities_v2_" +Util.getCurrentDateTime("yyyyMMdd") + ".db");
                }
            }

        } else if(argList.contains("Learn")) {

        } else if(argList.contains("Infer")) {
            String nonvisFile = ns.getString("nonvis_scores");
            String relationFile = ns.getString("relation_scores");
            String affinityFile = ns.getString("affinity_scores");
            String cardinalityFile = ns.getString("cardinality_scores");
            String typeCostFile = Overlord.resourcesDir + "hist_typePairLogProb.csv";

            //Set up the relation inference module
            RelationInference relInf = null;
            switch(ns.getString("inf_type")){
                case "relation": relInf = new RelationInference(docSet, nonvisFile,
                        relationFile, typeCostFile, null, null,
                        ns.getString("graph_root"), ns.getDouble("alpha"));
                    break;
                case "grounding": relInf = new RelationInference(docSet, nonvisFile,
                        null, null, affinityFile, cardinalityFile,
                        ns.getString("graph_root"), ns.getDouble("alpha"));
                    break;
                case "combined": relInf = new RelationInference(docSet, nonvisFile,
                        relationFile, typeCostFile, affinityFile, cardinalityFile,
                        ns.getString("graph_root"), ns.getDouble("alpha"));
                    break;
            }

            if(relInf == null){
                Logger.log(new Exception("Could not create relation inference module"));
                System.exit(1);
            }

            //Do inference
            relInf.infer(numThreads, ns.getBoolean("type_constraint"));

            //And evaluate it
            relInf.evaluate(ns.getBoolean("export_files"));
        }
	}

    /**Returns the Namespace object for the given parser, run over
     * the given args; Quits the application if arg parser is violated
     *
     * @param parser
     * @param args
     * @return
     */
    private static Namespace _parseArgs(ArgumentParser parser, String[] args)
    {
        //actually parse our args
        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch(ArgumentParserException apEx) {
            //Only print help if the exception wasn't a help
            //exception (I guess those are printed
            //automatically)
            if(!apEx.getClass().equals(HelpScreenException.class)) {
                parser.printHelp();
                if(apEx.getClass().equals(UnrecognizedArgumentException.class)) {
                    UnrecognizedArgumentException unArgEx =
                            (UnrecognizedArgumentException)apEx;
                    System.out.println("\n***ERROR*** Unrecognized argument: " +
                            unArgEx.getArgument());
                }
                else if(apEx.getMessage().contains("ambiguous option") ||
                        apEx.getMessage().contains("invalid choice") ||
                        apEx.getMessage().contains("is required") ||
                        apEx.getMessage().contains("unrecognized arguments") ||
                        apEx.getMessage().contains("too few arguments"))
                {
                    System.out.println("\n***ERROR*** " + apEx.getMessage());
                } else {
                    System.out.println(apEx.getMessage());
                    apEx.printStackTrace();
                }
            }
            System.exit(0);
        }
        return ns;
    }

    /**Sets up an flag argument, which stores true if specified
     *
     * @param arg
     * @param help
     */
    private static void _setArgument_flag(Argument arg, String help)
    {
        arg.help(help);
        arg.action(Arguments.storeTrue());
    }

    /**Sets up an option argument, allowing for one of opts options
     *
     * @param arg
     * @param opts
     * @param defaultVal
     * @param help
     */
    private static void _setArgument_opts(Argument arg, String[] opts,
                                          String defaultVal, String help)
    {
        arg.choices(Arrays.asList(opts));
        if(defaultVal != null)
            arg.setDefault(defaultVal);
        arg.help(help);
    }

    /**Sets up a string argument with the specified help text
     *
     * @param arg
     * @param help
     */
    private static void _setArgument(Argument arg, String help)
    {
        _setArgument(arg, help, null, null, null, null);
    }

    /**Sets up an argument of type with defaultVal and help
     *
     * @param arg
     * @param help
     * @param type
     * @param defaultVal
     */
    private static void _setArgument(Argument arg, String help,
                                     Class type, Object defaultVal)
    {
        _setArgument(arg, help, type, defaultVal, null, null);
    }

    /**Sets up a string argument with specified help and meta-var
     *
     * @param arg
     * @param help
     * @param metavar
     */
    private static void _setArgument(Argument arg, String help, String metavar)
    {
        _setArgument(arg, help, null, null, metavar, null);
    }

    /**Sets up required string argument with specified help and meta-var
     *
     * @param arg
     * @param help
     * @param metavar
     * @param required
     */
    private static void _setArgument(Argument arg, String help, String metavar, Boolean required)
    {
        _setArgument(arg, help, null, null, metavar, required);
    }

    /**Sets up a new argument with specified options
     *
     * @param arg
     * @param help
     * @param type
     * @param defaultVal
     * @param metavar
     * @param required
     */
    private static void _setArgument(Argument arg, String help,
                                     Class type, Object defaultVal, String metavar,
                                     Boolean required)
    {
        arg.help(help);
        if(metavar != null)
            arg.metavar(metavar);
        if(defaultVal != null)
            arg.setDefault(defaultVal);
        if(type != null)
            arg.type(type);
        if(required != null)
            arg.required(required);
    }

    public static boolean getConsoleConfirm() {
        System.out.print("Are you sure you want to continue (y/n)? ");
        char c = ' ';
        BufferedReader br =
                new BufferedReader(new InputStreamReader(System.in));
        try{
            String line = br.readLine();
            if(line.length() == 1)
                c = line.toCharArray()[0];
            while(c != 'y' && c != 'Y' && c != 'n' && c != 'N'){
                System.out.print("y or n, please: ");
                line = br.readLine();
                if(line.length() == 1)
                    c = line.toCharArray()[0];
            }
        } catch (IOException ioEx) {
            Logger.log(ioEx);
        }
        return c == 'y' || c == 'Y';
    }

}

/* Label Distro (1/18/2017)
 * 0,6122562
 * 1,997550
 * 2,105801
 * 3,105801
 */
