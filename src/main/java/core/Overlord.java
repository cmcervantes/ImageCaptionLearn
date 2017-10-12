package core;

import learn.*;
import nlptools.IllinoisAnnotator;
import nlptools.StanfordAnnotator;
import out.OutTable;
import structures.*;
import utilities.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import static core.DocumentLoader.getDocumentSet;

/**The Overlord is responsible for
 * parsing args and calling other PanOpt
 * modules
 * 
 * @author ccervantes
 */
public class Overlord 
{
    //Paths
    public static String dataPath = "/home/ccervan2/data/";
    public static String captionTePath = "/shared/projects/caption_te/";
    public static String wordnetDir = "/shared/data/WordNet-3.0/dict/";
    public static String word2vecPath = "/shared/projects/word2vec/word2vec.vector.gz";
    public static String boxFeatureDir = dataPath + "Flickr30kEntities_v1/box_feats/";

    //Dataset Paths
    public static String flickr30kPath = "/shared/projects/Flickr30kEntities_v2/";
    public static String flickr30kPath_legacy = "/shared/projects/Flickr30kEntities/";
    public static String[] flickr30k_mysqlParams =
            {"ccervan2.web.engr.illinois.edu", "ccervan2_root", "thenIdefyheaven!", "ccervan2_imageCaption"};
    public static String flickr30k_sqlite = flickr30kPath + "Flickr30kEntities_v2.db";
    public static String flickr30k_sqlite_legacy = flickr30kPath_legacy + "Flickr30kEntities_v1.db";
    public static String flickr30kResources = flickr30kPath + "resources/";
    public static String flickr30k_lexicon = "/shared/projects/Flickr30k/lexicons/";
    public static String mscocoPath = "/shared/projects/MSCOCO/";
    public static String[] mscoco_mysqlParams =
            {"ccervan2.web.engr.illinois.edu", "ccervan2_root", "thenIdefyheaven!", "ccervan2_coco"};
    public static String mscoco_sqlite = mscocoPath + "MSCOCO.db"; //old: 5/31; new: 6/14
    public static String mscocoResources = mscocoPath + "resources/";
    public static String mscoco_lexicon = mscocoResources + "coco_lex.csv";
    public static String snliPath = "/shared/projects/SNLI/";

	private static String _outroot;

	/**Main function. Parses <b>args</b> and calls
	 * other PanOpt modules
	 * 
	 * @param args - Command line arguments
	 */
	public static void main(String[] args)
	{
        //start our runtime clock
		Logger.startClock();

        //Set up the argument parser; add main args
        String desc = "ImageCaptionLearn has several " +
                "modules, each with their own args. To get " +
                "more info on a particular module, specify " +
                "the module with --help";
        ArgParser parser = new ArgParser("ImageCaptionLearn", desc);
        parser.setArgument_flag("--quiet", "Whether to log output");
        parser.setArgument("--log_delay",
                "Minimum seconds to wait between logging progress status messages",
                Integer.class, 90, null);
        parser.setArgument("--out",
                "Writes output to file with ROOT prefix", "ROOT", null);
        parser.setArgument("--threads", "Uses NUM threads, where applicable",
                Integer.class, 1, "NUM", false, null);
        String[] datasetOpts = {"flickr30k", "mscoco"};
        parser.setArgument_opts("--data", datasetOpts, "flickr30k",
                "Uses the specified dataset", null);
        String[] dataSplitOpts = {"train", "dev", "test", "all"};
        parser.setArgument_opts("--split", dataSplitOpts, null,
                "Loads data of the specified cross validation split (or all)", null);
        parser.setArgument_flag("--reviewed_only",
                "Loads only reviewed documents of the --data and --split");
        parser.setArgument("--rand_docs",
                "Loads NUM random documents of the given dataset and cross val split",
                Integer.class, null, "NUM", false, null);
        parser.setArgument_flag("--local", "Runs locally; queries mysqlDB");

        //Throwaway arguments will appear in the Debug module;
        //More permanent things will appear in their own modules
        parser.addSubParser("Debug");
        parser.setArgument_flag("--heterog_subset",
                "Exports a list of heterog subset pairs", "Debug");
        parser.setArgument_flag("--penult_filter",
                "Exports mention pairs for which the penult filter fires", "Debug");
        parser.setArgument_flag("--mod_subset",
                "Exports newly modified subset features", "Debug");

        /* Data Group */
        parser.addSubParser("Data");
        parser.setArgument("--convert_to_arff",
                "Converts the specified .feats file to .arff format",
                "PATH", "Data");
        String[] featOpts = {"relation", "affinity", "nonvis", "card"};
        parser.setArgument_opts("--extractFeats", featOpts, null,
                "Extracts features to --out", "Data");
        parser.setArgument_flag("--for_neural", "Whether extracted features are to be "+
                                "used in conjunction with word embeddings in a neural "+
                                "network (in practice, turns off various high-dim features)",
                                "Data");
        parser.setArgument_flag("--exclude_subset", "Whether to exclude the subset label "+
                "during relation feature extraction", "Data");
        parser.setArgument_flag("--exclude_partof", "Whether to exclude the partOf label "+
                "during relation feature extraction", "Data");
        parser.setArgument_flag("--include_cardinality", "Whether to include the true "+
                "cardinality (for train) or predicted cardinality distribution (for dev) "+
                "when generating nonvis/relation features; intended for use with "+
                "--cardinality_scores", "Data");
        parser.setArgument("--cardinality_scores",
                "Cardinality scores file; associates mention unique IDs with [0,1] "+
                "(0-10,11+) cardinality prediction", String.class, null,
                "FILE", false, "Data");
        String[] dbOpts = {"mysql", "sqlite"};
        parser.setArgument_opts("--buildDB", dbOpts, "mysql",
                "Rebuilds the specified database in the default locations", "Data");

        /* Learn Group */
        parser.addSubParser("Learn");
        parser.setArgument("--train_file", "Training data", "PATH", "Learn");
        parser.setArgument("--model_file", "Model file (save to / load from)", "PATH", "Learn");
        String[] learners = {"weka_multi", "liblinear_logistic"};
        parser.setArgument_opts("--learner", learners, "weka_multi",
                "Specifies which training is taking place", "Learn");
        parser.setArgument("--batch_size", "train arg; uses SIZE batches",
                Integer.class, 100, "SIZE", false, "Learn");
        parser.setArgument("--epochs", "train arg; run for NUM epochs",
                Integer.class, 1000, "NUM", false, "Learn");
        parser.setArgument("--eval_file", "Evaluation data", "PATH", "Learn");
        parser.setArgument_flag("--pronom_coref",
                "Evaluates rule-based pronominal coreference resolution", "Learn");

        /* Infer Group */
        parser.addSubParser("Infer");
        parser.setArgument("--nonvis_scores",
                "Nonvisual scores file; associates mention unique IDs with [0,1] nonvis prediction",
                "FILE", "Infer");
        parser.setArgument("--relation_scores",
                "Pairwise relation scores file; associates mention pair IDs with [0,1] (n,c,b,p) predictions",
                String.class, "/home/ccervan2/source/data/feats/flickr30kEntities_v2_relation_dev.scores", "FILE",
                false, "Infer");
        parser.setArgument("--affinity_scores",
                "Affinity scores file; associates mention|box unique IDs with [0,1] affinity prediction",
                String.class, "/home/ccervan2/source/data/feats/flickr30kEntities_v2_affinity_dev.scores",
                "FILE", false, "Infer");
        parser.setArgument("--cardinality_scores",
                "Cardinality scores file; associates mention unique IDs with [0,1] "+
                "(0-10,11+) cardinality prediction", String.class,
                "/home/ccervan2/source/data/feats/flickr30kEntities_v2_box_card_dev.scores",
                "FILE", false, "Infer");
        parser.setArgument_opts("--inf_type", new String[]{"relation", "grounding", "joint",
                                                           "joint_after_rel", "joint_after_grnd"},
                "relation", "Specified which inference module to use", "Infer");
        parser.setArgument_flag("--include_type_constraint", "Enables the inference type constraint", "Infer");
        parser.setArgument_flag("--exclude_subset", "Whether to exclude the subset label "+
                "during relation / joint inference", "Infer");
        parser.setArgument_flag("--exclude_box_exigence", "Whether to exclude the box exigence "+
                "constraint during grounding / joint inference", "Infer");
        parser.setArgument_flag("--export_files", "Writes examples to out/coref/htm/ and conll "+
                "files to out/coref/conll/", "Infer");
        parser.setArgument("--graph_root", "Loads previous inference "+
                "graphs from ROOT_relation.obj and ROOT_grounding.obj",
                "ROOT", "Infer");
        parser.setArgument("--alpha", "[0,1] alpha value",
                Double.class, 0.0, "NUM", false, "Infer");

        //Actually parse the arguments
        parser.parseArgs(args);

		//parse our main args
		if(!parser.getBoolean("quiet"))
			Logger.setVerbose();
        boolean runLocal = parser.getBoolean("local");
		_outroot = parser.getString("out");
        int numThreads = parser.getInt("threads");
        Logger.setStatusDelay(parser.getInt("log_delay"));
        String dataset = parser.getString("data");
        String split = parser.getString("split");
        Collection<Document> docSet = null;
        if(split != null){
            DBConnector conn = null;
            if(dataset.equals(datasetOpts[0]) && runLocal){
                conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1],
                                       flickr30k_mysqlParams[2], flickr30k_mysqlParams[3]);
            } else if(dataset.equals(datasetOpts[0])){
                conn = new DBConnector(flickr30k_sqlite);
            } else if(dataset.equals(datasetOpts[1]) && runLocal){
                conn = new DBConnector(mscoco_mysqlParams[0], mscoco_mysqlParams[1],
                                       mscoco_mysqlParams[2], mscoco_mysqlParams[3]);
            } else if(dataset.equals(datasetOpts[1])){
                conn = new DBConnector(mscoco_sqlite);
            }

            boolean reviewedOnly = parser.getBoolean("reviewed_only");
            Integer numRandImgs = parser.getInt("rand_docs");
            if(numRandImgs == null)
                numRandImgs = -1;

            int crossValFlag;
            switch(split){
                case "all": crossValFlag = -1; break;
                case "dev": crossValFlag = 0; break;
                case "train": crossValFlag = 1; break;
                case "test": crossValFlag = 2; break;
                default: crossValFlag = -1;
            }

            //get the documents
            docSet = getDocumentSet(conn, crossValFlag, reviewedOnly, numRandImgs);

            if(dataset.equals("mscoco")){
                Logger.log("WARNING: Adding Flickr30k Entities lexical types to COCO; fix this");
                Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscoco_lexicon);
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        m.setLexicalType(Mention.getLexicalEntry_flickr(m));
                    }
                }
            }
        }

        //Switch on the specified module, and parse module args
        List<String> argList = Arrays.asList(args);
        if(argList.contains("Debug")){
            if(parser.getBoolean("heterog_subset")) {
                Minion.export_subsetHeterogType(docSet);
            } else if(parser.getBoolean("penult_filter")){
                Minion.export_penultFilter(docSet);
            } else if(parser.getBoolean("mod_subset")){
                Minion.export_modSubsetFeats(docSet, split);
            } else {


                String raw_root = Overlord.dataPath + "tacl201711/raw/" +
                        dataset + "_" + split + "_" + Util.getCurrentDateTime("yyyyMMdd");

                //Preprocess.export_neuralCaptionFile(docList, raw_root);
                Preprocess.export_neuralRelationFiles(docSet, raw_root);
                //Preprocess.export_neuralNonvisFile(docList, raw_root);
                System.exit(0);


                /**Consider our label distribution when we have
                 * coref (symmetric)
                 * subset (directed; consistent)
                 * superset (directed; consistent)
                 * complement (symmetric; sets that belong to the same
                 *            hierarchy but are not otherwise directly related)
                 * intersect (symmetric)
                 * disjoint
                 */
                DoubleDict<String> newLabelDistro = new DoubleDict<>();
                for(Document d : docSet){
                    Map<String, Set<Integer>> chainBoxDict = new HashMap<>();
                    for(Chain c : d.getChainSet()){
                        chainBoxDict.put(c.getID(), new HashSet<>());
                        for(BoundingBox b : c.getBoundingBoxSet())
                            chainBoxDict.get(c.getID()).add(b.getIdx());
                    }
                    Set<Chain[]> subsetChainPairs = d.getSubsetChains();
                    Map<String, Set<String>> subsetDict = new HashMap<>();
                    for(Chain[] subsetPair : subsetChainPairs){
                        String subID = subsetPair[0].getID();
                        String supID = subsetPair[1].getID();
                        if(!subsetDict.containsKey(subID))
                            subsetDict.put(subID, new HashSet<>());
                        subsetDict.get(subID).add(supID);
                    }

                    List<Mention> mentions = d.getMentionList();
                    for(int i=0; i<mentions.size(); i++){
                        Mention m_i = mentions.get(i);
                        if(m_i.getChainID().equals("0"))
                            continue;

                        for(int j=i+1; j<mentions.size(); j++){
                            Mention m_j = mentions.get(j);
                            if(m_j.getChainID().equals("0"))
                                continue;

                            String chainID_i = m_i.getChainID();
                            String chainID_j = m_j.getChainID();
                            Set<String> supChains_i = subsetDict.get(chainID_i);
                            Set<String> supChains_j = subsetDict.get(chainID_j);
                            String label_ij = "disjoint", label_ji="disjoint";
                            if(chainID_i.equals(chainID_j)){
                                label_ij = "coref";
                                label_ji = "coref";
                            } else if(supChains_i != null && supChains_i.contains(chainID_j)){
                                label_ij = "subset";
                                label_ji = "supset";
                            } else if(supChains_j != null && supChains_j.contains(chainID_i)){
                                label_ji = "subset";
                                label_ij = "supset";
                            } else if(supChains_i != null && supChains_j != null){
                                Set<String> supIntersect = new HashSet<>(supChains_i);
                                supIntersect.retainAll(supChains_j);
                                if(!supIntersect.isEmpty()){
                                    label_ij = "complement";
                                    label_ji = "complement";
                                }
                            }
                            if(label_ij.equals( "disjoint")) {
                                Set<Integer> boxes_i = chainBoxDict.get(chainID_i);
                                Set<Integer> boxes_j = chainBoxDict.get(chainID_j);
                                Set<Integer> boxIntersect = new HashSet<>(boxes_i);
                                boxIntersect.retainAll(boxes_j);
                                if(!boxes_i.isEmpty() && !boxes_j.isEmpty() && !boxIntersect.isEmpty() &&
                                        boxes_i.size() != boxIntersect.size() && boxes_j.size() != boxIntersect.size()){
                                    if(m_i.getPronounType() == Mention.PRONOUN_TYPE.NONE &&
                                            m_j.getPronounType() == Mention.PRONOUN_TYPE.NONE){
                                        if(Mention.getLexicalTypeMatch(m_i, m_j) > 0){
                                            label_ij = "intersect";
                                            label_ji = "intersect";
                                        }
                                    } else {
                                        label_ij = "intersect";
                                        label_ji = "intersect";
                                    }
                                }
                            }
                            newLabelDistro.increment(label_ij);
                            newLabelDistro.increment(label_ji);
                        }
                    }
                }

                double ttl = newLabelDistro.getSum();
                for(String label : newLabelDistro.getSortedByValKeys(true)){
                    System.out.printf("%-10s: %d (%.2f%%)\n",
                                      label, (int)newLabelDistro.get(label),
                                      100.0 * newLabelDistro.get(label) / ttl);
                }




                /** How often is it the case that a chain has
                 *  more than one superset (excluding transitive
                 *  closure cases*/
                DoubleDict<Integer> subsetHist = new DoubleDict<>();
                for(Document d : docSet){
                    List<Chain[]> subsetChains =
                            new ArrayList<>(d.getSubsetChains());
                    int n_chains = -1;
                    Set<Integer> indicesToRemove = new HashSet<>();
                    while(n_chains != subsetChains.size()){
                        n_chains = subsetChains.size();
                        for(int i=0; i<n_chains; i++){
                            if(indicesToRemove.contains(i))
                                continue;
                            Chain[] pair_i = subsetChains.get(i);

                            for(int j=i+1; j<n_chains; j++){
                                if(indicesToRemove.contains(j))
                                    continue;
                                Chain[] pair_j = subsetChains.get(j);

                                for(int k=j+1; k<n_chains; k++){
                                    Chain[] pair_k = subsetChains.get(k);

                                    //Cases in which k is removed
                                    //i: ab; j: bc; k: ac
                                    //i: bc; j: ab; k: ac
                                    if(pair_i[1].equals(pair_j[0]) &&
                                       pair_j[1].equals(pair_k[1]) &&
                                       pair_i[0].equals(pair_k[0]) ||
                                       pair_i[1].equals(pair_k[1]) &&
                                       pair_j[1].equals(pair_i[0]) &&
                                       pair_j[0].equals(pair_k[0])){
                                        indicesToRemove.add(k);
                                    }
                                }
                            }
                        }
                        List<Chain[]> subsetChains_sansTrans = new ArrayList<>();
                        for(int i=0; i<n_chains; i++){
                            if(!indicesToRemove.contains(i))
                                subsetChains_sansTrans.add(subsetChains.get(i));
                        }
                        subsetChains = subsetChains_sansTrans;
                    }

                    //Now, subset chains contain only single-dimensional links
                    //where if a sub b and b sub c then the a sub c link is removed
                    Map<String, Set<String>> subSupDict = new HashMap<>();
                    for(Chain[] pair : subsetChains){
                        Chain sub = pair[0], sup = pair[1];
                        if(!subSupDict.containsKey(sub.getID()))
                            subSupDict.put(sub.getID(), new HashSet<>());
                        subSupDict.get(sub.getID()).add(sup.getID());
                    }
                    for(String subID : subSupDict.keySet())
                        subsetHist.increment(subSupDict.get(subID).size());
                }
                double totalSubsetChains = subsetHist.getSum();
                for(Integer numSup : subsetHist.getSortedByValKeys(true))
                    System.out.printf("%d: %d (%.3f%%)\n", numSup,
                            (int)subsetHist.get(numSup),
                            100.0 * subsetHist.get(numSup) / totalSubsetChains);

                System.exit(0);

                /*Preprocess.export_phraseLocalization_affinityLists(docSet, "trainAsDev", Overlord.dataPath +
                        "tacl201708/nn/bryan_affinity/" + dataset + "_trainAsDev");
                */
                /*
                Preprocess.export_relationLabelFile(docSet, Overlord.dataPath +
                        "tacl201708/nn/" + dataset + "_" + split + "_relationLabels");
                System.exit(0);*/

                /*
                Preprocess.export_neuralRelationFiles(docSet, Overlord.dataPath +
                    "tacl201708/nn/" + dataset + "_" + split);*/


                // Julia:   "What percentage of the intra-caption pronominal links
                //          in the training data have been manually corrected?"
                DoubleDict<String> intraLinkHist = new DoubleDict<>();
                for(Document d : docSet){
                    boolean isReviewed = d.reviewed;
                    for(Caption c : d.getCaptionList()){
                        List<Mention> mentions = c.getMentionList();
                        for(Mention m : mentions){
                            String label = isReviewed ? "reviewed" : "orig";
                            if(m.getPronounType() == Mention.PRONOUN_TYPE.NONE ||
                               m.getPronounType() == Mention.PRONOUN_TYPE.SEMI){
                                label = "nonpronom_" + label;
                            } else {
                                label = "pronom_" + label;
                            }
                            intraLinkHist.increment(label, mentions.size() - 1);
                        }
                    }
                }
                double totalIntraLinks = intraLinkHist.getSum();
                for(String key : intraLinkHist.keySet())
                    intraLinkHist.divide(key, totalIntraLinks);
                System.out.println(intraLinkHist);
                System.exit(0);


                Preprocess.export_phraseLocalization_affinityLists(docSet, "trainAsDev", Overlord.dataPath +
                        "tacl201708/nn/bryan_affinity/" + dataset + "_trainAsDev");
                System.exit(0);

                List<Document> docList = new ArrayList<>(docSet);
                Collections.shuffle(docList);
                docList = docList.subList(0, (int)(0.2*docSet.size()));
                raw_root += "_tune";

                Preprocess.export_neuralRelationFiles(docList.subList(0,
                        (int)(0.2*docSet.size())), Overlord.dataPath +
                        "tacl201708/nn/" + dataset + "_" + split + "_tune");
                System.exit(0);


                Set<String> stopWords =
                        new HashSet<>(FileIO.readFile_lineList(Overlord.flickr30kResources +
                                "stop_words.txt"));
                List<String> ll_normCaptions = new ArrayList<>();
                List<String> ll_mentionIndices = new ArrayList<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        List<String> tokenList = new ArrayList<>();
                        List<Integer> mentionIndices = new ArrayList<>();
                        for(Token t : c.getTokenList()){
                            String normText = t.toString().toLowerCase();
                            if(StringUtil.hasAlphaNum(normText) &&
                               !stopWords.contains(normText)){
                                tokenList.add(normText);
                                mentionIndices.add(t.mentionIdx);
                            }
                        }
                        ll_normCaptions.add(c.getUniqueID() + "\t" +
                            StringUtil.listToString(tokenList, " "));
                        ll_mentionIndices.add(c.getUniqueID() + "\t" +
                            StringUtil.listToString(mentionIndices, " "));
                    }
                }
                FileIO.writeFile(ll_normCaptions, Overlord.dataPath +
                        "tacl201708/nn/" + dataset + "_" + split + "_normCaps",
                        "txt", false);
                FileIO.writeFile(ll_mentionIndices, Overlord.dataPath +
                        "tacl201708/nn/" + dataset + "_" + split + "_mentionIndices",
                        "txt", false);

                System.exit(0);

/*
                ClassifyUtil.evaluateAffinity_coco(docSet, Overlord.dataPath + "tacl201708/scores/" +
                       //"mscoco_trainAsDev_coco30kCompleteModel_cocoTypes_affinity.scores",
                        "mscoco_trainAsDev_coco30kComplete_affinity.scores",
                        Overlord.dataPath + "tacl201708/scores/mscoco_train_nonvis.scores", false);

                System.exit(0);
*/


                if(split.equals("train") && dataset.equals("mscoco"))
                    split = "trainAsDev";
                Preprocess.export_phraseLocalization_ccaLists(docSet, split,
                        Overlord.dataPath + "tacl201708/cca/" + dataset + "_boxes/",
                        Overlord.dataPath + "tacl201708/cca/" + dataset + "_" + split);

                System.exit(0);




                String[] topTenNobox = {"street", "park", "water", "area", "beach",
                                        "city", "stage", "building", "game", "field"};
                Map<String, List<String>> exampleCaps = new HashMap<>();
                Map<String, List<Double>> docStatDict = new HashMap<>();
                for(String lemma : topTenNobox){
                    for(String label : new String[]{"nonvis", "nobox", "box"}){
                        exampleCaps.put(label + "_" + lemma, new ArrayList<>());
                        for(String boxLabel : new String[]{"boxes", "mentions", "capMentions"}){
                            docStatDict.put(label + "_" + lemma + "_" + boxLabel, new ArrayList<>());
                        }
                    }
                }

                for(Document d : docSet){
                    int totalBoxes = d.getBoundingBoxSet().size();
                    int totalMentions = d.getMentionList().size();


                    for(Caption c : d.getCaptionList()) {
                        int capMentions = c.getMentionList().size();

                        for (Mention m : c.getMentionList()) {
                            String normHead = m.getHead().getLemma().toLowerCase();
                            if(Arrays.asList(topTenNobox).contains(normHead)){
                                if (m.getChainID().equals("0")) {
                                    exampleCaps.get("nonvis_" + normHead).add(c.toString());
                                    docStatDict.get("nonvis_" + normHead + "_boxes").add(1.0*totalBoxes);
                                    docStatDict.get("nonvis_" + normHead + "_mentions").add(1.0*totalMentions);
                                    docStatDict.get("nonvis_" + normHead + "_capMentions").add(1.0*capMentions);
                                } else if (d.getBoxSetForMention(m).isEmpty()) {
                                    exampleCaps.get("nobox_" + normHead).add(c.toString());
                                    docStatDict.get("nobox_" + normHead + "_boxes").add(1.0*totalBoxes);
                                    docStatDict.get("nobox_" + normHead + "_mentions").add(1.0*totalMentions);
                                    docStatDict.get("nobox_" + normHead + "_capMentions").add(1.0*capMentions);
                                } else {
                                    exampleCaps.get("box_" + normHead).add(c.toString());
                                    docStatDict.get("box_" + normHead + "_boxes").add(1.0*totalBoxes);
                                    docStatDict.get("box_" + normHead + "_mentions").add(1.0*totalMentions);
                                    docStatDict.get("box_" + normHead + "_capMentions").add(1.0*capMentions);
                                }
                            }
                        }
                    }
                }

                for(String lemma : topTenNobox){
                    for(String label : new String[]{"nonvis", "nobox", "box"}){
                        for(String boxLabel : new String[]{"boxes", "mentions", "capMentions"}){
                            String group = label + "_" + lemma + "_" + boxLabel;
                            System.out.printf("\t %s: %.2f\n", group, StatisticalUtil.getMean(docStatDict.get(group)));
                        }
                    }
                }
                System.exit(0);

                List<String> ll_topTen = new ArrayList<>();
                for(String lemma : topTenNobox){
                    String[] labels = {"nonvis", "nobox", "box"};
                    for(String label : labels){
                        String groupName = label + "_" + lemma;
                        ll_topTen.add("");
                        ll_topTen.add("--------------- " + groupName + " ---------------");

                        List<String> caps = exampleCaps.get(groupName);
                        Collections.shuffle(caps);
                        for(int i=0; i<Math.min(caps.size(), 50); i++)
                            ll_topTen.add(caps.get(i));
                    }
                }
                FileIO.writeFile(ll_topTen, "ex_topTenNobox", "txt", true);

                System.exit(0);


                Map<String, DoubleDict<String>> histDict = new HashMap<>();
                String[] histLabels = {"nonvis", "nobox", "box", "nonvis_heads", "nobox_heads", "box_heads"};
                for(String histLabel : histLabels)
                    histDict.put(histLabel, new DoubleDict<>());

                Map<String, Set<String>> mentionCapDict = new HashMap<>();
                for (Document d : docSet) {
                    for (Mention m : d.getMentionList()) {
                        String normText = m.toString().toLowerCase();
                        String normHead = m.getHead().getLemma().toLowerCase();

                        if (!mentionCapDict.containsKey(normText))
                            mentionCapDict.put(normText, new HashSet<>());
                        mentionCapDict.get(normText).add(d.getID() + "#" + m.getCaptionIdx());

                        if (m.getChainID().equals("0")) {
                            histDict.get("nonvis").increment(normText);
                            histDict.get("nonvis_heads").increment(normHead);
                        } else if (d.getBoxSetForMention(m).isEmpty()){
                            histDict.get("nobox").increment(normText);
                            histDict.get("nobox_heads").increment(normHead);
                        } else {
                            histDict.get("box").increment(normText);
                            histDict.get("box_heads").increment(normHead);
                        }
                    }
                }

                Map<String, String> captionDict = new HashMap<>();
                for (Document d : docSet)
                    for (int i = 0; i < d.getCaptionList().size(); i++)
                        captionDict.put(d.getID() + "#" + i, d.getCaptionList().get(i).toString());

                for(String histLabel : histLabels){
                    double histTotal = histDict.get(histLabel).getSum();
                    List<String> ll = new ArrayList<>();
                    for(String normText : histDict.get(histLabel).getSortedByValKeys(true)){
                        ll.add(String.format("%s --- %d (%.2f%%)",
                                normText, (int)histDict.get(histLabel).get(normText),
                                100.0 * histDict.get(histLabel).get(normText) / histTotal));
                    }
                    String filename, ext;

                    if(histLabel.contains("heads")){
                        filename = "hist_" + histLabel;
                        ext = "csv";
                    } else {
                        for(String normText : histDict.get(histLabel).getSortedByValKeys(true)){
                            ll.add(String.format("%s --- %d (%.2f%%)",
                                    normText, (int)histDict.get(histLabel).get(normText),
                                    100.0 * histDict.get(histLabel).get(normText) / histTotal));
                        }
                        for(String normText : histDict.get(histLabel).getSortedByValKeys(true)){
                            ll.add(String.format("%s --- %d (%.2f%%)",
                                    normText, (int)histDict.get(histLabel).get(normText),
                                    100.0 * histDict.get(histLabel).get(normText) / histTotal));
                            for(String capID : mentionCapDict.get(normText))
                                ll.add("\t" + captionDict.get(capID));
                        }
                        filename = "ex_" + histLabel + "Context";
                        ext = "txt";
                    }
                    FileIO.writeFile(ll, filename, ext, true);
                }
                System.exit(0);


                ClassifyUtil.evaluateNonvis(docSet, Overlord.dataPath + "tacl201708/scores/");



                DoubleDict<String> labelHist = new DoubleDict<>();
                for(Document d : docSet){
                    Set<String> subsetMentions = d.getSubsetMentions();
                    List<Mention> mentions = d.getMentionList();
                    for(int i=0; i<mentions.size(); i++){
                        Mention m_i = mentions.get(i);
                        if(m_i.getChainID().equals("0"))
                            continue;

                        for(int j=i+1; j<mentions.size(); j++){
                            Mention m_j = mentions.get(j);
                            if(m_j.getChainID().equals("0"))
                                continue;

                            String id_ij = Document.getMentionPairStr(m_i, m_j);
                            String id_ji = Document.getMentionPairStr(m_j, m_i);

                            if(m_i.getChainID().equals(m_j.getChainID()))
                                labelHist.increment("coref", 2);
                            else if(subsetMentions.contains(id_ij))
                                labelHist.increment("subset");
                            else if(subsetMentions.contains(id_ji))
                                labelHist.increment("supset");
                            else
                                labelHist.increment("null", 2);
                        }
                    }
                }
                double totalLabels = labelHist.getSum();
                for(String key : labelHist.keySet())
                    labelHist.divide(key, totalLabels);
                System.out.println(labelHist);
                System.exit(0);






                Preprocess.export_categories(docSet);

                System.exit(0);


                List<String> ll_affinity = new ArrayList<>();
                for(Document d : docSet){
                    List<Mention> mentions = d.getMentionList();
                    Set<BoundingBox> boxes = d.getBoundingBoxSet();
                    for(Mention m : mentions){
                        String mentionCats = Mention.getLexicalEntry_cocoCategory(m, true);
                        for(BoundingBox b : boxes){
                            String id = m.getUniqueID() + "|" + b.getUniqueID();
                            double pred_1 = 0.0000000000000000000000001;
                            double pred_0 = 0.9999999999999999999999999;
                            if(mentionCats != null && mentionCats.contains(b.getCategory())){
                                double tmp = pred_1;
                                pred_1 = pred_0;
                                pred_0 = tmp;
                            }
                            ll_affinity.add(id + "," + Math.log(pred_0) + "," + Math.log(pred_1));
                        }
                    }
                }
                FileIO.writeFile(ll_affinity, dataset + "_" + split + "_heur_affinity", "csv", false);
                System.exit(0);



//000000233079.jpg#1;mention:1|000000233079.jpg;box:15,-0.0308530183369,-3.49390754091



                ClassifyUtil.exportStanfordCorefConll(docSet);

                //ClassifyUtil.evaluateAffinity_coco(docSet, Overlord.dataPath +
                //        "tacl201708/scores/mscoco_dev_30k_affinity.scores",
                //        Overlord.dataPath + "tacl201708/scores/mscoco_dev_nonvis.scores", false);

                //ClassifyUtil.evaluateNonvis(docSet, Overlord.dataPath + "tacl201708/scores/flickr30k_test_nonvis.scores");


                /*
                Preprocess.export_phraseLocalization_ccaLists(docSet, "dev", "flickr30k",
                        Overlord.dataPath + "tacl201708/cca/mscoco_boxes/",
                        Overlord.dataPath + "tacl201708/cca/mscoco_dev");*/


                /*
                Preprocess.export_phraseLocalization_convertBoxFeats(docSet,
                        Overlord.dataPath + "tacl201708/cca/mscoco_dev_imData_feats.csv",
                        Overlord.dataPath + "tacl201708/cca/mscoco_boxes/");*/


                /**/

                //ClassifyUtil.evaluateNonvis(docSet, Overlord.dataPath +
                //        "tacl201708/scores/" + dataset + "_" + split + "_nonvis.scores");
                //ClassifyUtil.exportStanfordCorefConll(docSet);
                System.exit(0);


                DoubleDict<Integer> nonvisLabelDict = new DoubleDict<>();
                List<String> ll_feats = FileIO.readFile_lineList("/home/ccervan2/data/tacl201708/feats/coco30k_trainDev_nonvis.feats");
                for(String fvStr : ll_feats){
                    FeatureVector fv = FeatureVector.parseFeatureVector(fvStr);
                    nonvisLabelDict.increment((int)fv.label);
                }
                System.out.print(nonvisLabelDict);
                System.exit(0);




                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()) {
                        int chunkIdx_max = -1;
                        for (Chunk ch : c.getChunkList())
                            if (ch.getIdx() > chunkIdx_max)
                                chunkIdx_max = ch.getIdx();
                        if(chunkIdx_max >= c.getChunkList().size())
                            System.out.println(c.getUniqueID());
                    }
                }
                System.exit(0);

                List<String> allowedConjList =
                        Arrays.asList(",", "and", ", and", "on and");

                Map<String, Set<Caption>> partOfDict = new HashMap<>();
                partOfDict.put("xofy", new HashSet<>());
                partOfDict.put("agent_list", new HashSet<>());
                partOfDict.put("bodypart_list", new HashSet<>());
                partOfDict.put("clothing_list", new HashSet<>());
                partOfDict.put("no_list", new HashSet<>());
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        List<Mention> agents = new ArrayList<>();
                        List<Mention> bodyparts = new ArrayList<>();
                        List<Mention> clothing = new ArrayList<>();

                        for(int i=0; i<c.getMentionList().size(); i++){
                            Mention m = c.getMentionList().get(i);

                            switch(m.getLexicalType()){
                                case "people":
                                case "animals": agents.add(m);
                                    break;
                                case "bodyparts": bodyparts.add(m);
                                    break;
                                case "clothing":
                                case "colors":
                                case "clothing/colors": clothing.add(m);
                                    break;
                                default:
                                    if(m.getPronounType().isAnimate())
                                        agents.add(m);
                            }
                        }

                        if(agents.isEmpty() || (bodyparts.isEmpty() && clothing.isEmpty()))
                            continue;

                        for(int i=1; i<c.getMentionList().size(); i++){
                            Mention m = c.getMentionList().get(i);
                            Mention m_j = c.getMentionList().get(i-1);
                            List<Token> interstitialTokens = c.getInterstitialTokens(m_j, m);
                            if(!agents.isEmpty() && agents.get(agents.size()-1).equals(m) &&
                                    interstitialTokens.size() == 1 && interstitialTokens.get(0).toString().equals("of")){
                                if(!bodyparts.isEmpty() && bodyparts.get(bodyparts.size()-1).equals(m_j))
                                    partOfDict.get("xofy").add(c);
                                if(!clothing.isEmpty() && clothing.get(clothing.size()-1).equals(m_j))
                                    partOfDict.get("xofy").add(c);
                            }
                        }

                        for(int i=1; i<agents.size(); i++){
                            Mention m = agents.get(i);
                            Mention m_j = agents.get(i-1);
                            List<Token> interstitialTokens = c.getInterstitialTokens(m_j, m);
                            String interStr = StringUtil.listToString(interstitialTokens, " ").toLowerCase().trim();
                            if(allowedConjList.contains(interStr))
                                partOfDict.get("agent_list").add(c);
                        }
                        for(int i=1; i<bodyparts.size(); i++){
                            Mention m = bodyparts.get(i);
                            Mention m_j = bodyparts.get(i-1);
                            List<Token> interstitialTokens = c.getInterstitialTokens(m_j, m);
                            String interStr = StringUtil.listToString(interstitialTokens, " ").toLowerCase().trim();
                            if(allowedConjList.contains(interStr))
                                partOfDict.get("bodypart_list").add(c);
                        }
                        for(int i=1; i<clothing.size(); i++){
                            Mention m = clothing.get(i);
                            Mention m_j = clothing.get(i-1);
                            List<Token> interstitialTokens = c.getInterstitialTokens(m_j, m);
                            String interStr = StringUtil.listToString(interstitialTokens, " ").toLowerCase().trim();
                            if(allowedConjList.contains(interStr))
                                partOfDict.get("clothing_list").add(c);
                        }

                        boolean addedCap = false;
                        for(String caseStr : partOfDict.keySet()){
                            addedCap |= partOfDict.get(caseStr).contains(c);
                        }
                        if(!addedCap)
                            partOfDict.get("no_list").add(c);
                    }
                }


                for(String caseStr : partOfDict.keySet()){
                    System.out.println(caseStr + ": " + partOfDict.get(caseStr).size());
                    List<String> captions = new ArrayList<>();
                    for(Caption c : partOfDict.get(caseStr))
                        captions.add(c.getUniqueID() + "\t" + c.toString());
                    FileIO.writeFile(captions, "ex_" + caseStr + "_captions", "txt", true);
                }


                /*
                Preprocess.export_phraseLocalization_ccaLists(docSet, "trainAsDev",
                        "flickr30k", "/home/ccervan2/data/tacl201708/mscoco_boxes/",
                        "/home/ccervan2/data/tacl201708/mscoco_trainAsDev");*/
                System.exit(0);



                Mention.initializeLexicons(Overlord.flickr30k_lexicon, null);
                Caption.initLemmatizer();
                Cardinality.initCardLists(Overlord.flickr30kResources + "collectiveNouns.txt");
                DocumentLoader.exportCOCOFiles(docSet, "/shared/projects/Flickr30kEntities_v2/flickr30kEntities_v2");
                System.exit(0);


                List<String> ll_biff = new ArrayList<>();
                for(Document d : docSet){
                    boolean hasBowlCupWineglass = false;
                    boolean hasXofY = false;
                    for(Mention m : d.getMentionList())
                        if(m.toString().toLowerCase().contains(" of "))
                            hasXofY = true;
                    for(BoundingBox b : d.getBoundingBoxSet())
                        if(b.getCategory().equals("cup") || b.getCategory().equals("bowl") || b.getCategory().equals("wineglass"))
                            hasBowlCupWineglass = true;

                    if(hasBowlCupWineglass && hasXofY)
                        ll_biff.add("'" + d.getID() + "'");
                }
                System.out.println("WHERE img_id IN (" + StringUtil.listToString(ll_biff, ", ") + ")");
                System.exit(0);

                Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscocoResources + "coco_lex.csv");
                int numMentions = 0;
                int numNonvis = 0;
                int numNobox = 0;
                DoubleDict<String> supercat_nobox = new DoubleDict<>();
                DoubleDict<String> supercat_nobox_fallback = new DoubleDict<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        numMentions++;

                        if(m.getChainID().equals("0"))
                            numNonvis++;

                        Set<BoundingBox> boxSet = d.getBoxSetForMention(m);
                        if(!m.getChainID().equals("0") && boxSet.isEmpty()){
                            numNobox++;
                            String cocoCat = Mention.getLexicalEntry_cocoCategory(m, false);
                            String cocoCat_fallback = Mention.getLexicalEntry_cocoCategory(m, true);
                            if(cocoCat != null)
                                supercat_nobox.increment(Mention.getSuperCategory(cocoCat));
                            if(cocoCat_fallback != null)
                                for(String cat : cocoCat_fallback.split("/"))
                                    supercat_nobox_fallback.increment(Mention.getSuperCategory(cat));
                        }
                    }
                }
                System.out.printf("Mentions: %d; Nonvis: %d (%.2f%%); Vis Nobox: %d (%.2f%%)\n",
                        numMentions, numNonvis, 100.0 * numNonvis / numMentions, numNobox,
                        100.0 * numNobox / numMentions);
                System.out.printf("Strict coco cat nobox: %d (%.2f%%); Fallback nobox: %d (%.2f%%)\n",
                        (int)supercat_nobox.getSum(), 100.0 * supercat_nobox.getSum() / numMentions,
                        (int)supercat_nobox_fallback.getSum(),
                        100.0 * supercat_nobox_fallback.getSum() / numMentions);
                System.out.println("-------Strict Category--------");
                System.out.print(supercat_nobox.toString());
                System.out.println("-------Fallback Category--------");
                System.out.print(supercat_nobox_fallback.toString());
                System.exit(0);






                Set<String> corefPairs = new HashSet<>();
                Set<String> subsetPairs = new HashSet<>();
                Set<String> partOfPairs = new HashSet<>();
                int mentionPairs = 0;
                for(Document d : docSet){
                    subsetPairs.addAll(d.getSubsetMentions());
                    partOfPairs.addAll(d.getPartOfMentions());
                    for(int i=0; i<d.getMentionList().size(); i++){
                        Mention m_i = d.getMentionList().get(i);
                        for(int j=i+1; j<d.getMentionList().size(); j++){
                            mentionPairs += 2;
                            Mention m_j = d.getMentionList().get(j);
                            if(!m_i.getChainID().equals("0") &&
                               m_i.getChainID().equals(m_j.getChainID())){
                                corefPairs.add(Document.getMentionPairStr(m_i, m_j));
                                corefPairs.add(Document.getMentionPairStr(m_j, m_i));
                            }
                        }
                    }
                }
                System.out.printf("Coref: %d (%.2f%%)\n", corefPairs.size(), 100.0 * corefPairs.size() / mentionPairs);
                System.out.printf("Subst: %d (%.2f%%)\n", subsetPairs.size(), 100.0 * subsetPairs.size() / mentionPairs);
                System.out.printf("Partf: %d (%.2f%%)\n", partOfPairs.size(), 100.0 * partOfPairs.size() / mentionPairs);
                System.exit(0);




                for(Document d : docSet){
                    if(d.getID().equals("000000487060.jpg")){
                        //000000140922.jpg
                        //000000266912.jpg
                        //000000526534.jpg
                        for(String partOfPair : d.getPartOfMentions()){
                            Mention[] pair = d.getMentionPairFromStr(partOfPair);
                            System.out.println("-----");

                            String coarseType_i = null, coarseType_j = null;
                            switch(pair[0].getLexicalType()){
                                case "people":
                                case "animals": coarseType_i = "agents";
                                    break;
                                case "bodyparts": coarseType_i = "bodyparts";
                                    break;
                                case "clothing":
                                case "colors":
                                case "clothing/colors": coarseType_i = "clothing";
                                    break;
                                default:
                                    if(pair[0].getPronounType().isAnimate())
                                        coarseType_i = "agents";
                            }
                            switch(pair[1].getLexicalType()){
                                case "people":
                                case "animals": coarseType_j = "agents";
                                    break;
                                case "bodyparts": coarseType_j = "bodyparts";
                                    break;
                                case "clothing":
                                case "colors":
                                case "clothing/colors": coarseType_j = "clothing";
                                    break;
                                default:
                                    if(pair[1].getPronounType().isAnimate())
                                        coarseType_j = "agents";
                            }

                            System.out.println(coarseType_i + "\t" + coarseType_j);
                            System.out.println(pair[0].getLexicalType() + "\t" + pair[1].getLexicalType());
                            System.out.println(pair[0].getPronounType() + "\t" + pair[1].getPronounType());
                            System.out.println(pair[0].getChainID() + "\t" + pair[1].getChainID());
                            System.out.println(pair[0].toString() + "\t" + pair[1].toString());
                        }
                    }
                }




                System.exit(0);

                List<String> partOfExamples = new ArrayList<>();
                for(Document d : docSet){
                    Set<Chain[]> partOfChains = d.getPartOfChains();
                    if(!partOfChains.isEmpty()){
                        for(Caption c : d.getCaptionList())
                            partOfExamples.add(c.toString());
                        for(Chain[] pair : partOfChains)
                            partOfExamples.add("\t{" +
                                    StringUtil.listToString(pair[0].getMentionSet(), "|") + "}\t{" +
                                    StringUtil.listToString(pair[1].getMentionSet(), "|") + "}");
                    }
                }
                FileIO.writeFile(partOfExamples, "ex_partOf", "txt", true);
                System.exit(0);


                DoubleDict<String> hist_heur = new DoubleDict<>(), hist_intr = new DoubleDict<>();
                Map<Mention, Set<Mention>> heur_cloth = new HashMap<>(), heur_part = new HashMap<>(),
                        intr_cloth = new HashMap<>(), intr_part = new HashMap<>();
                DoubleDict<String> hist_interstitial = new DoubleDict<>();

                //get the heuristic-attached cloth and bodyparts
                Map<Mention, AttrStruct> attrDict =
                        ClassifyUtil.attributeAttachment_agent(docSet);
                for(Mention m_agent : attrDict.keySet()){
                    AttrStruct as = attrDict.get(m_agent);
                    Set<Mention> clothing = new HashSet<>(), bodyparts = new HashSet<>();
                    for(Mention m_attr : as.getAttributeMentions()){
                        String lexType = m_attr.getLexicalType();
                        if(lexType.contains("bodyparts"))
                            bodyparts.add(m_attr);
                        else if(lexType.contains("clothing") || lexType.contains("colors"))
                            clothing.add(m_attr);
                    }
                    if(!clothing.isEmpty()){
                        if(!heur_cloth.containsKey(m_agent))
                            heur_cloth.put(m_agent, new HashSet<>());
                        heur_cloth.get(m_agent).addAll(clothing);
                    }
                    if(!bodyparts.isEmpty()){
                        if(!heur_part.containsKey(m_agent))
                            heur_part.put(m_agent, new HashSet<>());
                        heur_part.get(m_agent).addAll(bodyparts);
                    }

                    for(Mention m : clothing){
                        String pair_code;
                        if(m_agent.getIdx() < m.getIdx())
                            pair_code = "agent|cloth";
                        else
                            pair_code = "cloth|agent";
                        hist_heur.increment(pair_code);
                    }
                    for(Mention m : bodyparts){
                        String pair_code;
                        if(m_agent.getIdx() < m.getIdx())
                            pair_code = "agent|part";
                        else
                            pair_code = "part|agent";
                        hist_heur.increment(pair_code);
                    }
                }

                //Get the agent/bodyparts and agent/clothing that are separated by
                //verbs (subj/obj or interstitial) or preps (interstitial)
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        //get subject/object pairings
                        Map<Mention, Chunk> mentionVerbDict_subj = new HashMap<>();
                        Map<Mention, Chunk> mentionVerbDict_obj = new HashMap<>();
                        for(Mention m : c.getMentionList()){
                            Chunk verb_subj = c.getSubjectOf(m);
                            Chunk verb_obj = c.getObjectOf(m);
                            if(verb_subj != null)
                                mentionVerbDict_subj.put(m, verb_subj);
                            if(verb_obj != null)
                                mentionVerbDict_obj.put(m, verb_obj);
                        }
                        Map<Chunk, Set<Mention>> verbMentionDict_subj =
                                Util.invertMap(mentionVerbDict_subj);
                        Map<Chunk, Set<Mention>> verbMentionDict_obj =
                                Util.invertMap(mentionVerbDict_obj);
                        for(Chunk verb_subj : verbMentionDict_subj.keySet()){
                            Set<Mention> subjOfMentions = verbMentionDict_subj.get(verb_subj);
                            if(verbMentionDict_obj.containsKey(verb_subj)){
                                Set<Mention> objOfMentions = verbMentionDict_obj.get(verb_subj);
                                for(Mention m_subj : subjOfMentions){
                                    String lexType_subj = m_subj.getLexicalType();
                                    for(Mention m_obj : objOfMentions) {
                                        String lexType_obj = m_obj.getLexicalType();

                                        boolean agent_subj = false, agent_obj=false,
                                                part_subj = false, part_obj = false,
                                                cloth_subj = false, cloth_obj = false;
                                        if(m_subj.getPronounType().isAnimate() || lexType_subj.equals("people") ||
                                                lexType_subj.equals("animals"))
                                            agent_subj = true;
                                        if(m_obj.getPronounType().isAnimate() || lexType_obj.equals("people") ||
                                                lexType_obj.equals("animals"))
                                            agent_obj = true;
                                        if(lexType_subj.equals("bodyparts"))
                                            part_subj = true;
                                        if(lexType_obj.equals("bodyparts"))
                                            part_obj = true;
                                        if(lexType_subj.equals("clothing") || lexType_subj.equals("colors") || lexType_subj.equals("clothing/colors"))
                                            cloth_subj = true;
                                        if(lexType_obj.equals("clothing") || lexType_obj.equals("colors") || lexType_obj.equals("clothing/colors"))
                                            cloth_obj = true;

                                        //determine what kind of pairing this is
                                        String pair_code = null;
                                        if(agent_subj){
                                            if(part_obj){
                                                if(!intr_part.containsKey(m_subj))
                                                    intr_part.put(m_subj, new HashSet<>());
                                                intr_part.get(m_subj).add(m_obj);
                                                if(m_subj.getIdx() < m_obj.getIdx())
                                                    pair_code = "agent|part";
                                                else
                                                    pair_code = "part|agent";
                                            } else if(cloth_obj){
                                                if(!intr_cloth.containsKey(m_subj))
                                                    intr_cloth.put(m_subj, new HashSet<>());
                                                intr_cloth.get(m_subj).add(m_obj);
                                                if(m_subj.getIdx() < m_obj.getIdx())
                                                    pair_code = "agent|cloth";
                                                else
                                                    pair_code = "cloth|agent";
                                            }
                                        } else if (agent_obj) {
                                            if(part_subj){
                                                if(!intr_part.containsKey(m_obj))
                                                    intr_part.put(m_obj, new HashSet<>());
                                                intr_part.get(m_obj).add(m_subj);
                                                if(m_obj.getIdx() < m_subj.getIdx())
                                                    pair_code = "agent|part";
                                                else
                                                    pair_code = "part|agent";
                                            } else if(cloth_subj){
                                                if(!intr_cloth.containsKey(m_obj))
                                                    intr_cloth.put(m_obj, new HashSet<>());
                                                intr_cloth.get(m_obj).add(m_subj);
                                                if(m_obj.getIdx() < m_subj.getIdx())
                                                    pair_code = "agent|cloth";
                                                else
                                                    pair_code = "cloth|agent";
                                            }
                                        }

                                        if(pair_code != null) {
                                            hist_intr.increment(pair_code);
                                            hist_interstitial.increment(verb_subj.getChunkType());
                                            hist_interstitial.increment(verb_subj.toString().toLowerCase());
                                        }
                                    }
                                }
                            }
                        }

                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            Mention m_i = mentions.get(i);
                            String lexType_i = m_i.getLexicalType();
                            for(int j=i+1; j<mentions.size(); j++){
                                Mention m_j = mentions.get(j);
                                String lexType_j = m_j.getLexicalType();
                                List<Chunk> intrsttl = c.getInterstitialChunks(m_i, m_j);
                                if(intrsttl.size() == 1){
                                    String pair_case = "";
                                    if(lexType_i.equals("people") || lexType_i.equals("animals") || m_i.getPronounType().isAnimate()){
                                        if(lexType_j.equals("bodyparts"))
                                            pair_case = "agent|part";
                                        else if(lexType_j.equals("clothing") || lexType_j.equals("colors") || lexType_j.equals("clothing/colors"))
                                            pair_case = "agent|cloth";
                                    } else if(lexType_j.equals("people") || lexType_j.equals("animals") || m_j.getPronounType().isAnimate()){
                                        if(lexType_i.equals("bodyparts"))
                                            pair_case = "part|agent";
                                        else if(lexType_i.equals("clothing") || lexType_i.equals("colors") || lexType_i.equals("clothing/colors"))
                                            pair_case = "cloth|agent";
                                    }

                                    if(pair_case.equals("agent|part")){
                                        if(!intr_part.containsKey(m_i))
                                            intr_part.put(m_i, new HashSet<>());
                                        if(!intr_part.get(m_i).contains(m_j)){
                                            intr_part.get(m_i).add(m_j);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    } else if(pair_case.equals("part|agent")){
                                        if(!intr_part.containsKey(m_j))
                                            intr_part.put(m_j, new HashSet<>());
                                        if(!intr_part.get(m_j).contains(m_i)){
                                            intr_part.get(m_j).add(m_i);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    } else if(pair_case.equals("agent|cloth")){
                                        if(!intr_cloth.containsKey(m_i))
                                            intr_cloth.put(m_i, new HashSet<>());
                                        if(!intr_cloth.get(m_i).contains(m_j)){
                                            intr_cloth.get(m_i).add(m_j);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    } else if(pair_case.equals("cloth|agent")){
                                        if(!intr_cloth.containsKey(m_j))
                                            intr_cloth.put(m_j, new HashSet<>());
                                        if(!intr_cloth.get(m_j).contains(m_i)){
                                            intr_cloth.get(m_j).add(m_i);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Logger.log("Heuristic Attachment");
                System.out.println(hist_heur);
                Logger.log("Verb/Interstitial Attachment");
                System.out.println(hist_intr);
                FileIO.writeFile(hist_interstitial, "hist_interstitial_attatch", "csv", true);


                Set<String> pairIDs_parts_heur = new HashSet<>(), pairIDs_cloth_heur = new HashSet<>(),
                        pairIDs_parts_intr = new HashSet<>(), pairIDs_cloth_intr = new HashSet<>();
                for(Mention agent : heur_cloth.keySet())
                    for(Mention cloth : heur_cloth.get(agent))
                        pairIDs_cloth_heur.add(Document.getMentionPairStr(agent, cloth));
                for(Mention agent : heur_part.keySet())
                    for(Mention part : heur_part.get(agent))
                        pairIDs_parts_heur.add(Document.getMentionPairStr(agent, part));
                for(Mention agent : intr_cloth.keySet())
                    for(Mention cloth : intr_cloth.get(agent))
                        pairIDs_cloth_intr.add(Document.getMentionPairStr(agent, cloth));
                for(Mention agent : intr_part.keySet())
                    for(Mention part : intr_part.get(agent))
                        pairIDs_parts_intr.add(Document.getMentionPairStr(agent, part));

                Logger.log("Heuristic Pairs");
                System.out.printf("cloth: %d; parts: %d\n", pairIDs_cloth_heur.size(), pairIDs_parts_heur.size());
                Logger.log("Interstitial Pairs");
                System.out.printf("cloth: %d; parts: %d\n", pairIDs_cloth_intr.size(), pairIDs_parts_intr.size());
                Logger.log("Intersection");
                Set<String> intersect_cloth = new HashSet<>(pairIDs_cloth_heur);
                intersect_cloth.retainAll(pairIDs_cloth_intr);
                Set<String> intersect_parts = new HashSet<>(pairIDs_parts_heur);
                intersect_parts.retainAll(pairIDs_parts_intr);
                System.out.printf("cloth: %d; parts: %d\n", intersect_cloth.size(), intersect_parts.size());



                System.exit(0);

/*
                Mention.initializeLexicons(Overlord.flickr30k_lexicon, null);
                Caption.initLemmatizer();
                Cardinality.initCardLists(Overlord.flickr30kResources + "collectiveNouns.txt");
                DBConnector conn = new DBConnector(mscoco_mysqlParams[0], mscoco_mysqlParams[1],
                                mscoco_mysqlParams[2], mscoco_mysqlParams[3]);
                docSet = DocumentLoader.getDocumentSet(conn);

                int numReviewed = 0;
                for(Document d : docSet)
                    if(d.reviewed)
                        numReviewed++;
                System.out.println(numReviewed);


                Set<String> docIDs_anno = new HashSet<>(FileIO.readFile_lineList("coco_annotation_20170529.txt"));
                for(Document d : docSet)
                    if(docIDs_anno.contains(d.getID()))
                        d.reviewed = true;

                String query_mention = "UPDATE mention SET lexical_type=?, card_str=? "+
                                       "WHERE img_id=? AND caption_idx=? AND mention_idx=?;";
                String query_chain = "INSERT INTO chain(img_id, chain_id, assoc_box_ids) "+
                                     "VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE "+
                                     "assoc_box_ids=assoc_box_ids;";
                Set<Object[]> paramSet_mention = new HashSet<>();
                Set<Object[]> paramSet_chain = new HashSet<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList())
                        paramSet_mention.add(new Object[]{m.getLexicalType(), m.getCardinality().toString(),
                            d.getID(), m.getCaptionIdx(), m.getIdx()});
                    for(Chain c : d.getChainSet()){
                        List<String> boxIDs = new ArrayList<>();
                        for(BoundingBox b : c.getBoundingBoxSet())
                            boxIDs.add(String.valueOf(b.getIdx()));
                        Collections.sort(boxIDs);
                        paramSet_chain.add(new Object[]{d.getID(), c.getID(), StringUtil.listToString(boxIDs, "|")});
                    }
                }

                try{
                    conn.update(query_mention, paramSet_mention, 50000, 8);
                    conn.update(query_chain, paramSet_chain, 50000, 8);
                } catch(Exception ex){
                    Logger.log(ex);
                }
                System.exit(0);*/
                //Minion.importCocoData(Overlord.mscocoPath + "coco_caps_20170531.coref");
                //System.exit(0);
/*
                Mention.initializeLexicons(Overlord.flickr30k_lexicon, null);
                Caption.initLemmatizer();
                Cardinality.initCardLists(Overlord.flickr30kResources +
                        "collectiveNouns.txt");
/*
                /*

                Logger.log("Loading Alice's MSCOCO captions");
                Set<Caption> captions = new HashSet<>();
                for(String line : FileIO.readFile_lineList("/shared/projects/DenotationGraph/corpora/MSCOCO/MSCOCO.coref")){
                    try {
                        captions.add(Caption.fromCorefStr(line));
                    } catch(Exception ex){
                        Logger.log(ex);
                    }
                }

                Logger.log("Applying XofY fixes");
                Set<Caption> captions_mod = Minion.applyXofYFixes(captions);

                Logger.log("Writing new captions to .coref file");
                List<String> ll_newCaps = new ArrayList<>();
                captions_mod.forEach(c -> ll_newCaps.add(c.toCorefString(true)));
                FileIO.writeFile(ll_newCaps,
                        Overlord.mscocoPath + "coco_caps", "coref", true);

                */



                StanfordAnnotator stanfordAnno = StanfordAnnotator.createCoreference(false);

                /*
                for(Document d : docSet){
                    if(d.getID().equals("541046380.jpg")){
                        String text = "";
                        for(Caption c : d.getCaptionList())
                            text += c.toString() + " ";
                        text = text.trim();
                        Document d_stanford = stanfordAnno.annotate(d.getID(), text);
                        for(Caption c : d_stanford.getCaptionList()) {
                            System.out.println(c.toCorefString(false));
                            for(Token t : c.getTokenList()){
                                System.out.println(t.toString() + " (" + t.chainID + ")");
                            }
                        }
                        System.out.println("---");
                        for(Chain c : d_stanford.getChainSet()){
                            System.out.println(c.toDebugString());
                            for(Mention m : c.getMentionSet()) {
                                System.out.println("\t"+m.toDebugString());
                                for(Token t : m.getTokenList()){
                                    System.out.println("\t\t"+t.toDebugString());
                                }
                            }
                        }
                        System.exit(0);
                    }
                }*/


                System.exit(0);

                Minion.export_cocoCategoryStats_coverage(docSet);
                Minion.export_cocoCategoryStats_givenBox(docSet);
                Minion.export_cocoCategoryStats_givenMention(docSet);
                System.exit(0);


                Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscocoResources + "coco_lex.csv");
                Set<String> supercategories = Mention.getCOCOSupercategories();
                Set<String> categories = Mention.getCOCOCategories();

                DoubleDict<String> peopleDistro = new DoubleDict<>();
                DoubleDict<String> flickrPeopleCocoHist = new DoubleDict<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        peopleDistro.increment("mention_count");

                        String category = Mention.getLexicalEntry_cocoCategory(m, true);
                        String type = Mention.getLexicalEntry_flickr(m);
                        boolean cocoPeople = category != null && category.contains("person");
                        boolean flickrPeople = type != null && type.contains("people");

                        if(cocoPeople)
                            peopleDistro.increment("coco_people");
                        if(flickrPeople)
                            peopleDistro.increment("flickr_people");
                        if(cocoPeople && flickrPeople)
                            peopleDistro.increment("coco_flickr_people");

                        if(category == null && flickrPeople) {
                            peopleDistro.increment("flickr_people_coco_none");
                            flickrPeopleCocoHist.increment(m.toString().toLowerCase());
                        }
                    }
                }

                System.out.printf("%.2f%% mentions with coco people\n",
                        100.0 * peopleDistro.get("coco_people") /
                                peopleDistro.get("mention_count"));
                System.out.printf("%.2f%% mentions with flickr people\n",
                        100.0 * peopleDistro.get("flickr_people") /
                                peopleDistro.get("mention_count"));
                System.out.printf("%.2f%% of coco people covered by flickr lex\n",
                        100.0 * peopleDistro.get("coco_flickr_people") /
                                peopleDistro.get("coco_people"));
                System.out.printf("%.2f%% of flickr people covered by coco lex\n",
                        100.0 * peopleDistro.get("coco_flickr_people") /
                                peopleDistro.get("flickr_people"));
                System.out.printf("%.2f%% of mentions that are flickr people and uncovered coco\n",
                        100.0 * peopleDistro.get("flickr_people_coco_none") /
                                peopleDistro.get("mention_count"));
                FileIO.writeFile(flickrPeopleCocoHist, "hist_uncoveredCocoFlickrPeople", "csv", true);

                System.exit(0);

/*
                Map<String, Caption> origCapDict = new HashMap<>();
                for(Caption c : origCaps)
                    origCapDict.put(c.getUniqueID(), c);
                Map<String, Caption> newCapDict = new HashMap<>();
                for(Caption c : caps_new)
                    newCapDict.put(c.getUniqueID(), c);
                List<String> ll_capDiff = new ArrayList<>();
                for(String capID : origCapDict.keySet()){
                    Caption c_orig = origCapDict.get(capID);
                    String capStrOrig = c_orig.toCorefString(false).trim();
                    if(!newCapDict.containsKey(capID)){
                        Logger.log("ERROR: couldn't find " + capID);
                        continue;
                    }
                    Caption c_new = newCapDict.get(capID);
                    String capStrNew = c_new.toCorefString(false).trim();
                    if(!capStrOrig.equals(capStrNew)){
                        ll_capDiff.add("orig: " + capStrOrig);
                        ll_capDiff.add("new : " + capStrNew);
                        for(int i=0; i<c_orig.getTokenList().size(); i++){
                            Token t_orig = c_orig.getTokenList().get(i);
                            Token t_new = c_new.getTokenList().get(i);
                            ll_capDiff.add("token: " + t_orig.toString() +
                                           ";orig_chunk:" + t_orig.chunkIdx +
                                           ";orig_mention:" + t_orig.mentionIdx +
                                           ";new_chunk:" + t_new.chunkIdx +
                                           ";new_mention:" + t_new.mentionIdx);
                        }
                    }
                }
                FileIO.writeFile(ll_capDiff, "diff_cocoXofY", "txt", true);
                //Minion.importCocoData("/shared/projects/DenotationGraph/corpora/MSCOCO/MSCOCO.coref");
                System.exit(0);
                */



                /*
                IllinoisTagger tggr_biff = new IllinoisTagger(Overlord.dataPath + "pos/");
                List<String> cocoLines_orig =
                        FileIO.readFile_lineList(Overlord.mscocoPath + "coco_caps.txt");
                List<String> cocoLines_new = new ArrayList<>();
                for(String line : cocoLines_orig){
                    String[] lineParts = line.split("\t");
                    edu.illinois.cs.cogcomp.lbjava.nlp.seg.Token[] toks = tggr_biff.predict(lineParts[1]);
                    StringBuilder sb = new StringBuilder();
                    sb.append(lineParts[0]);
                    sb.append("\t");
                    for(int i=0; i<toks.length; i++){
                        sb.append(toks[i].form);
                        if(i < toks.length-1)
                            sb.append(" ");
                    }
                    cocoLines_new.add(sb.toString());
                }
                FileIO.writeFile(cocoLines_new, "coco_caps", "token", false);
                System.exit(0);*/

                //Mention.initLexiconDict(Overlord.flickr30k_lexicon);
                /*
                Caption.initLemmatizer();
                Cardinality.initCardLists(Overlord.flickr30kResources + "collectiveNouns.txt");
                String posDir = Overlord.dataPath + "pos/";
                String chunkDir = Overlord.dataPath + "chunk/";
                String cocoData = Overlord.mscocoPath + "coco_caps.txt";
                Map<String, Caption[]> captionDict = Minion.parseCocoCaptions(cocoData, posDir, chunkDir);

                Map<String, Caption> newCaps = new HashMap<>();
                for(Caption[] capArr : captionDict.values())
                    for(Caption c : capArr)
                        if(c != null)
                            newCaps.put(c.getUniqueID(), c);

                Map<String, Caption> xofyCaps = new HashMap<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        boolean isXofyCap = false;
                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            if(i < mentions.size() - 1){
                                List<Token> intrToks = c.getInterstitialTokens(mentions.get(i), mentions.get(i+1));
                                if(intrToks.size() == 1 && intrToks.get(0).toString().equals("of"))
                                    isXofyCap = true;
                            }
                        }
                        if(isXofyCap)
                            xofyCaps.put(c.getUniqueID(), c);
                    }
                }
                List<String> xofyCapIDs = new ArrayList<>(xofyCaps.keySet());
                Collections.sort(xofyCapIDs);
                List<String> xofyCaps_diff = new ArrayList<>();
                for(String ID : xofyCapIDs) {
                    Caption c = xofyCaps.get(ID);
                    if(newCaps.containsKey(ID)){
                        String capStr_orig = c.toCorefString(false).trim();
                        String capStr_new = newCaps.get(ID).toCorefString(false).trim();
                        if(!capStr_orig.equals(capStr_new)){
                            xofyCaps_diff.add("orig: " + capStr_orig);
                            xofyCaps_diff.add("new : " + capStr_new);
                        }
                    }
                }
                FileIO.writeFile(xofyCaps_diff, "diff_cocoXofYCaps", "txt", true);
                System.exit(0);*/

                System.exit(0);


                String[] notableCats = {"handbag", "dining table", "potted plant",
                                        "backpack", "baseball glove", "spoon",
                                        "cup", "knife", "fork", "bottle", "oven"};
                /*
                for(String cat : notableCats){
                    List<String> ll_caps = new ArrayList<>();
                    for(Caption c : zeroMentionCaps.get(cat))
                        ll_caps.add(c.toString());
                    FileIO.writeFile(ll_caps, "out/charts/zeroMentionCaps_" + cat, "txt", true);
                }


                Map<String, DoubleDict<String>> cocoFreqDict = new HashMap<>();
                for(String cat : superCats.keySet())
                    cocoFreqDict.put(cat, new DoubleDict<>());


                for(Document d : docSet){
                    Set<String> categories = new HashSet<>();
                    d.getBoundingBoxSet().forEach(b -> categories.add(b.getCategory()));
                    for(Mention m : d.getMentionList()){
                        String head = m.getHead().getLemma().toLowerCase();
                        String lastTwo = "";
                        List<Token> toks = m.getTokenList();
                        if(toks.size() > 1)
                            lastTwo = toks.get(toks.size()-2).getLemma() + " ";
                        lastTwo += toks.get(toks.size()-1).getLemma();
                        lastTwo = lastTwo.toLowerCase();

                        //Increment the covered cases
                        boolean foundCat = false, foundFallback = false;
                        for(String cat : categories){
                            if(catHeads.get(cat).contains(lastTwo) || catHeads.get(cat).contains(head)){
                                cocoFreqDict.get(cat).increment(cat.toUpperCase());
                                foundCat = true;
                                //We could break here, but there shoudn't be overlap with this
                            }
                        }

                        //Increment the fallback cases
                        if(!foundCat){
                            for(String cat : categories){
                                if(catFallbacks.get(cat).contains(lastTwo) || catFallbacks.get(cat).contains(head)){
                                    cocoFreqDict.get(cat).increment("FALLBACK");
                                    foundFallback = true;
                                    //Dont break here, since we want to count each of the multiple fallbacks
                                }
                            }
                        }

                        //Increment the strings themselves if they're not even fallback options
                        if(!foundCat && !foundFallback){
                            for(String cat : categories){
                                if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                                    cocoFreqDict.get(cat).increment("PRONOUN");
                                else
                                    cocoFreqDict.get(cat).increment(lastTwo);
                            }
                        }
                    }
                }

                OutTable ot_cocoFreq = new OutTable("category", "penultimate", "freq");
                for(String cat : cocoFreqDict.keySet()){
                    List<String> sortedPenults = cocoFreqDict.get(cat).getSortedByValKeys(true);
                    sortedPenults = sortedPenults.subList(0, Math.min(sortedPenults.size(), 100));
                    for(String penult : sortedPenults)
                        ot_cocoFreq.addRow(cat, penult, cocoFreqDict.get(cat).get(penult));
                }
                ot_cocoFreq.writeToCsv("hist_coco_cat_penults", true);
                System.exit(0);
*/

                //Minion.export_bryanPreproc_coco(docSet);
                //System.exit(0);
                Mention.initializeLexicons(flickr30k_lexicon, null);
                List<String> ll_snli = FileIO.readFile_lineList(Overlord.snliPath +
                        "snli_1.0_train.jsonl");
                IllinoisAnnotator annotator =
                        IllinoisAnnotator.createChunker(Overlord.dataPath + "pos/",
                                Overlord.dataPath + "chunk/");
                Set<SNLIPair> snliPairs = new HashSet<>();
                for(String line : ll_snli)
                    snliPairs.add(new SNLIPair(line, annotator));
                List<String> ll_premise = new ArrayList<>(), ll_hyp = new ArrayList<>();
                for(SNLIPair pair : snliPairs){
                    String pairID = pair.getID();
                    ll_premise.add(pairID + "\t" + pair.getPremise().toCorefString());
                    ll_hyp.add(pairID + "\t" + pair.getHypothesis().toCorefString());
                }
                FileIO.writeFile(ll_premise, Overlord.snliPath + "snli_train_premise.coref");
                FileIO.writeFile(ll_hyp, Overlord.snliPath + "snli_train_hypothesis.coref");
                System.exit(0);



                Map<String, String> xTypeDict = new HashMap<>();
                String[][] xofyTable =
                        FileIO.readFile_table(Overlord.dataPath + "hist_xofy_lemma_coco_20170424.csv");
                for(int i=1; i<xofyTable.length; i++){
                    String[] row = xofyTable[i];
                    if(!row[0].isEmpty() && !row[1].isEmpty())
                        xTypeDict.put(row[0].replace("\"", ""), row[1].replace("\"", ""));
                }
                Set<String> xTypes = new HashSet<>();
                for(String xLemma : xTypeDict.keySet())
                    xTypes.add(xTypeDict.get(xLemma));
                for(String xType : xTypes)
                    System.out.println(xType);
                System.exit(0);

                Map<String, DoubleDict<String>> xofYLemmas = new HashMap<>();

                DoubleDict<String> xLemmaHist = new DoubleDict<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            Mention m_i = mentions.get(i);
                            Mention m_j = null;
                            if(i < mentions.size() - 1)
                                m_j = mentions.get(i+1);

                            String xLemma = null, yLemma = null;
                            if(m_i.toString().toLowerCase().contains(" of ")){
                                List<Token> xToks = new ArrayList<>();
                                for(Token t : m_i.getTokenList()){
                                    if(t.toString().toLowerCase().equals("of"))
                                        break;
                                    else
                                        xToks.add(t);
                                }
                                xLemma = xToks.get(xToks.size() - 1).getLemma().toLowerCase();
                                yLemma = m_i.getHead().getLemma().toLowerCase();
                            } else if(m_j != null){
                                List<Token> intrsttlToks = c.getInterstitialTokens(m_i, m_j);
                                if(intrsttlToks.size() == 1 && intrsttlToks.get(0).toString().toLowerCase().equals("of")){
                                    xLemma = m_i.getHead().getLemma().toLowerCase();
                                    yLemma = m_j.getHead().getLemma().toLowerCase();
                                }
                            }

                            //if this is an XofY construction (or X in XofY), log it
                            if(xLemma != null){
                                xLemmaHist.increment(xLemma);
                                if(!xofYLemmas.containsKey(xLemma))
                                    xofYLemmas.put(xLemma, new DoubleDict<>());
                                xofYLemmas.get(xLemma).increment(yLemma);
                            }
                        }
                    }
                }


                double totalXofY = xLemmaHist.getSum();
                OutTable ot_xofy_coco = new OutTable("x_lemma", "category", "freq", "perc", "top_y_heads");
                for(String xLemma : xLemmaHist.getSortedByValKeys(true)){
                    double xFreq = xLemmaHist.get(xLemma);
                    String xType = "unk";
                    if(xTypeDict.containsKey(xLemma))
                        xType = xTypeDict.get(xLemma);
                    List<String> yLemmaFreqs = new ArrayList<>();
                    for(String yLemma : xofYLemmas.get(xLemma).getSortedByValKeys(true))
                        yLemmaFreqs.add(String.format("%s (%.2f%%)", yLemma,
                                xofYLemmas.get(xLemma).get(yLemma) / xFreq));
                    ot_xofy_coco.addRow(xLemma, xType, (int)xFreq, xFreq / totalXofY,
                            StringUtil.listToString(yLemmaFreqs, " | "));
                }
                ot_xofy_coco.writeToCsv("hist_xofy_lemmas_coco", true);
                System.exit(0);


                /*
                List<Double> boxCounts = new ArrayList<>();
                for(Document d : docSet)
                    boxCounts.add((double)d.getBoundingBoxSet().size());
                System.out.println(StatisticalUtil.getMean(boxCounts));
                */
                //DBConnector conn = new DBConnector("COCO_" + Util.getCurrentDateTime("yyyyMMdd") + ".db");
               //DBConnector conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1],
               //        flickr30k_mysqlParams[2], "ccervan2_coco");

                /*
                Collection<Document> docSet_coco =
                        DocumentLoader.getDocumentSet(Overlord.dataPath + "mscoco/coco_train_sub_20170331.coref",
                                Overlord.flickr30k_lexicon, Overlord.flickr30kResources);
                double mentionCount = 0.0;
                for(Document d : docSet_coco)
                    mentionCount += d.getMentionList().size();
                System.out.println(docSet_coco.size());
                System.out.println(mentionCount / (5*docSet_coco.size()));
                System.exit(0);*/







                /*
                Set<String> imgIDs = new HashSet<>(); imgIDs.add("76739724.jpg");
                DBConnector conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1], flickr30k_mysqlParams[2],flickr30k_mysqlParams[3]);
                docSet = DocumentLoader.getDocumentSet(conn, imgIDs);*/


                //Minion.export_attachmentCases(docSet);
                System.exit(0);
                /*
                DoubleDict<String> clothFreq = new DoubleDict<>();
                DoubleDict<String> partFreq = new DoubleDict<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        List<Mention> agentList = new ArrayList<>();
                        List<Mention> partList = new ArrayList<>();
                        List<Mention> clothList = new ArrayList<>();
                        for(Mention m : c.getMentionList()){
                            if(m.getLexicalType().equals("people") || m.getLexicalType().equals("animals") ||
                               m.getPronounType().isAnimate()){
                                agentList.add(m);
                            } else if(m.getLexicalType().equals("bodyparts")){
                                partList.add(m);
                            } else if(m.getLexicalType().equals("clothing") || m.getLexicalType().equals("colors") ||
                                    m.getLexicalType().equals("clothing/colors")){
                                clothList.add(m);
                            }
                        }
                        List<List<Mention>> agentClusters =
                                ClassifyUtil.collapseMentionListToConstructionList(agentList, c);
                        List<List<Mention>> clothClusters =
                                ClassifyUtil.collapseMentionListToConstructionList(clothList, c);
                        if(partList.size() > 0)
                            partFreq.increment(agentClusters.size() + "|" + partList.size());
                        if(clothClusters.size() > 0)
                            clothFreq.increment(agentClusters.size() + "|" + clothClusters.size());
                    }
                }
                FileIO.writeFile(partFreq, "hist_parts", "csv", true);
                FileIO.writeFile(clothFreq, "hist_cloths", "csv", true);
                System.exit(0);*/

                /*
                Set<String> docIDs = new HashSet<>();
                docIDs.add("4059698218.jpg");
                DBConnector conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1], flickr30k_mysqlParams[2],flickr30k_mysqlParams[3]);
                docSet = DocumentLoader.getDocumentSet(conn, docIDs);
*/
                /*
                DoubleDict<String> clothHeads = new DoubleDict<>();
                DoubleDict<String> partHeads = new DoubleDict<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        if(!m.getChainID().equals("0")){
                            if(m.getLexicalType().contains("clothing"))
                                clothHeads.increment(m.getHead().getLemma().toLowerCase());
                            if(m.getLexicalType().contains("bodyparts"))
                                partHeads.increment(m.getHead().getLemma().toLowerCase());
                        }
                    }
                }
                FileIO.writeFile(clothHeads, "hist_clothHead", "csv", false);
                FileIO.writeFile(partHeads, "hist_bodypartHead", "csv", false);
                System.exit(0);*/





            }
        } else if(argList.contains("Data")) {
            String featsFileToConvert = parser.getString("convert_to_arff");
            String featsToExtract = parser.getString("extractFeats");
            String buildDB = parser.getString("buildDB");


            if(featsFileToConvert != null) {
                WekaMulticlass.exportToArff(featsFileToConvert);
            } else if(featsToExtract != null){
                if(featsToExtract.equals("relation")){
                    ClassifyUtil.exportFeatures_relation(docSet, _outroot, numThreads,
                            parser.getBoolean("for_neural"),
                            !parser.getBoolean("exclude_subset"),
                            !parser.getBoolean("exclude_partof"),
                            parser.getBoolean("include_cardinality"),
                            parser.getString("cardinality_scores"));
                }
                else if(featsToExtract.equals("affinity")) {
                    ClassifyUtil.exportFeatures_affinity(docSet, split);
                } else if(featsToExtract.equals("nonvis")) {
                    ClassifyUtil.exportFeatures_nonvis(docSet, _outroot,
                            parser.getBoolean("for_neural"),
                            parser.getBoolean("include_cardinality"),
                            parser.getString("cardinality_scores"));
                } else if(featsToExtract.equals("card")) {
                    ClassifyUtil.exportFeatures_cardinality(docSet, _outroot,
                            parser.getBoolean("for_neural"));
                }
            } else if(buildDB != null){
                System.out.println("WARNING: there's a bug where certain cardinalities are null");
                if(buildDB.equals("mysql")){
                    Preprocess.buildImageCaptionDB(Overlord.flickr30kPath + "Flickr30kEntities_v2.coref",
                            Overlord.flickr30kPath + "RELEASE/", Overlord.flickr30kResources + "img_comments.csv",
                            Overlord.flickr30kResources + "img_crossval.csv", Overlord.flickr30kResources + "img_reviewed.txt",
                            flickr30k_mysqlParams[0], flickr30k_mysqlParams[1], flickr30k_mysqlParams[2], flickr30k_mysqlParams[3]);
                } else if(buildDB.equals("sqlite")){
                    Preprocess.buildImageCaptionDB(Overlord.flickr30kPath + "Flickr30kEntities_v2.coref",
                            Overlord.flickr30kPath + "RELEASE/", Overlord.flickr30kResources + "img_comments.csv",
                            Overlord.flickr30kResources + "img_crossval.csv", Overlord.flickr30kResources + "img_reviewed.txt",
                            Overlord.flickr30kPath + "Flickr30kEntities_v2_" +Util.getCurrentDateTime("yyyyMMdd") + ".db");
                }
            }
        } else if(argList.contains("Learn")) {

        } else if(argList.contains("Infer")) {
            String nonvisFile = parser.getString("nonvis_scores");
            String relationFile = parser.getString("relation_scores");
            String affinityFile = parser.getString("affinity_scores");
            String cardinalityFile = parser.getString("cardinality_scores");

            //Set up the relation inference module
            ILPInference inf = null;
            ILPInference.InferenceType infType =
                    ILPInference.InferenceType.valueOf(parser.getString("inf_type").toUpperCase());
            switch(infType){
                case RELATION: inf = new ILPInference(docSet, infType, nonvisFile, relationFile,
                        null, null, parser.getString("graph_root"),
                        parser.getBoolean("include_type_constraint"),
                        parser.getBoolean("exclude_box_exigence"),
                        parser.getBoolean("exclude_subset"));
                    break;
                case GROUNDING: inf = new ILPInference(docSet, infType, nonvisFile, null,
                        affinityFile, cardinalityFile, parser.getString("graph_root"),
                        parser.getBoolean("include_type_constraint"),
                        parser.getBoolean("exclude_box_exigence"),
                        parser.getBoolean("exclude_subset"));
                    break;
                case JOINT:
                case JOINT_AFTER_REL:
                case JOINT_AFTER_GRND:
                        inf = new ILPInference(docSet, infType, nonvisFile, relationFile,
                        affinityFile, cardinalityFile, parser.getString("graph_root"),
                        parser.getBoolean("include_type_constraint"),
                        parser.getBoolean("exclude_box_exigence"),
                        parser.getBoolean("exclude_subset"));
                    break;
            }


            if(inf== null){
                Logger.log(new Exception("Could not create relation inference module"));
                System.exit(1);
            }

            //Do inference
            inf.infer(numThreads);

            //And evaluate it
            inf.evaluate(parser.getBoolean("export_files"));
        }
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
