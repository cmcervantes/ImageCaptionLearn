package core;

import learn.ClassifyUtil;
import learn.ILPInference;
import learn.Preprocess;
import learn.WekaMulticlass;
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
    public static String denotation_sqlite = "/shared/projects/DenotationDB/results_20160121.db";


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
        parser.setArgument_opts("--neuralPreproc",
                new String[]{"caption", "nonvis", "relation", "card", "affinity"},
                null, "Exports neural preprocessing files to --out", "Data");
        parser.setArgument_opts("--ccaPreproc",
                new String[]{"box_files", "box_feats", "cca_lists"},
                null, "Exports cca preproeccsing files to --out", "Data");
        parser.setArgument("--boxFeatDir", "Box feature directory, used for --ccaPreproc", "Data");
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
                String.class, null, "FILE", false, "Infer");
        parser.setArgument("--affinity_scores",
                "Affinity scores file; associates mention|box unique IDs with [0,1] affinity prediction",
                String.class, null, "FILE", false, "Infer");
        parser.setArgument("--cardinality_scores",
                "Cardinality scores file; associates mention unique IDs with [0,1] "+
                "(0-10,11+) cardinality prediction", String.class, null, "FILE", false, "Infer");
        parser.setArgument_opts("--inf_type", new String[]{"visual", "relation", "grounding",
                "visual_relation", "visual_grounding", "relation_grounding",
                "visual_relation_grounding", "grounding_then_relation",
                "grounding_then_visual_relation", "relation_then_grounding",
                "relation_then_visual_grounding"}, null,
                "Specifies which inference module to use", "Infer");
        parser.setArgument_flag("--include_type_constraint", "Enables the inference type constraint", "Infer");
        parser.setArgument_flag("--exclude_subset", "Whether to exclude the subset label "+
                "during relation / joint inference", "Infer");
        parser.setArgument_flag("--exclude_box_exigence", "Whether to exclude the box exigence "+
                "constraint during grounding / joint inference", "Infer");
        parser.setArgument_flag("--only_keep_positive_links", "Whether to keep only positive links "+
                "during sequential inference modes (e.g. grounding_then_relation", "Infer");
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

                ClassifyUtil.evaluateAffinity_cocoHeuristic(docSet);
                System.exit(0);

                Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscoco_lexicon);

                ScoreDict<Integer> grndScoreDict = new ScoreDict<>();
                for(Document d : docSet){
                    Map<String, Set<BoundingBox>> catBoxes = new HashMap<>();
                    for(BoundingBox b : d.getBoundingBoxSet()){
                        String cat = b.getCategory();
                        if(!catBoxes.containsKey(cat))
                            catBoxes.put(cat, new HashSet<>());
                        catBoxes.get(cat).add(b);
                    }

                    for(Mention m : d.getMentionList()){
                        String catStr = Mention.getLexicalEntry_cocoCategory(m);
                        Set<String> cats = new HashSet<>();
                        if(catStr != null)
                            cats.addAll(Arrays.asList(catStr.split("/")));

                        Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                        for(BoundingBox b : d.getBoundingBoxSet()){
                            int gold = assocBoxes.contains(b) ? 1 : 0;
                            int pred = 0;
                            String cat_b = b.getCategory();
                            if(catBoxes.get(cat_b).size() == 1 &&
                               cats.size() > 0 && cats.contains(cat_b)){
                                pred = 1;
                            }
                            grndScoreDict.increment(gold, pred);
                        }
                    }
                }
                grndScoreDict.printCompleteScores();;
                System.exit(0);




                /* How often is it the case that coco coreference chains
                 * do not share a category?
                 */
                DoubleDict<String> cocoCatStat = new DoubleDict<>();
                double ttlChain = 0.0;
                for(Document d : docSet){
                    for(Chain c : d.getChainSet()){
                        if(c.getMentionSet().size() == 0)
                            continue;
                        ttlChain++;

                        List<Mention> mentions = new ArrayList<>(c.getMentionSet());
                        boolean foundNoCat = false;
                        boolean foundCat = false;
                        boolean foundDisjointPair = false;
                        boolean foundMismatchPair = false;
                        Set<String> catIntersect = new HashSet<>();
                        for(int i=0; i<mentions.size(); i++){
                            Mention m_i = mentions.get(i);
                            String cat_i = Mention.getLexicalEntry_cocoCategory(m_i);
                            Set<String> cats_i = new HashSet<>();
                            if(cat_i != null){
                                foundCat = true;
                                cats_i.addAll(Arrays.asList(cat_i.split("/")));
                            } else {
                                foundNoCat = true;
                            }

                            if(i == 0){
                                catIntersect.addAll(cats_i);
                            } else {
                                catIntersect.retainAll(cats_i);
                            }

                            for(int j=i+1; j<mentions.size(); j++){
                                Mention m_j = mentions.get(j);
                                String cat_j = Mention.getLexicalEntry_cocoCategory(m_j);
                                Set<String> cats_j = new HashSet<>();
                                if(cat_j != null)
                                    cats_j.addAll(Arrays.asList(cat_j.split("/")));

                                Set<String> ijIntersect = new HashSet<>(cats_i);
                                ijIntersect.retainAll(cats_j);
                                if(cats_i.size() > 0 && cats_j.size() > 0 && ijIntersect.size() == 0)
                                    foundDisjointPair = true;
                                if(cat_i != null && cat_j != null && !cat_i.equals(cat_j))
                                    foundMismatchPair = true;
                            }
                        }

                        if(foundNoCat)
                            cocoCatStat.increment("has_m_with_nocat");
                        else
                            cocoCatStat.increment("all_m_have_cats");
                        if(foundCat)
                            cocoCatStat.increment("has_m_with_cat");
                        else
                            cocoCatStat.increment("no_m_has_cat");
                        if(foundCat && foundNoCat)
                            cocoCatStat.increment("mixed_cat_nocat");

                        if(foundDisjointPair)
                            cocoCatStat.increment("has_disjoint_m_pair");
                        if(foundMismatchPair)
                            cocoCatStat.increment("has_mismatch_m_pair");
                        if(!foundNoCat && !foundMismatchPair)
                            cocoCatStat.increment("all_same_cat");

                        if(!foundNoCat && catIntersect.size() > 0)
                            cocoCatStat.increment("contains_common_cat");
                    }
                }
                for(String k : cocoCatStat.keySet())
                    cocoCatStat.divide(k, ttlChain / 100.0);
                System.out.println(cocoCatStat);

                System.exit(0);

                Logger.log("Loading scores");
                List<String> ll_affScores = FileIO.readFile_lineList("/home/ccervan2/data/tacl201801/bryan/chris_scores.csv");
                Logger.log("Loading IDs");
                List<String> ll_affIDs = FileIO.readFile_lineList("/home/ccervan2/data/tacl201711/neural_affinity/flickr30k_test_id.txt");
                Logger.log("Mapping scores to IDs");
                Map<String, Double> affPred = new HashMap<>();
                for(int i=0; i<ll_affIDs.size(); i++){
                    double predScore = 1.0 / (1.0 + Math.exp(-Double.parseDouble(ll_affScores.get(i))));
                    affPred.put(ll_affIDs.get(i).trim(), predScore);
                }
                Logger.log("Evaluating affinity");
                ScoreDict<Integer> affScores = new ScoreDict<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                        for(BoundingBox b : d.getBoundingBoxSet()){
                            int gold = !assocBoxes.isEmpty() && assocBoxes.contains(b) ? 1 : 0;
                            String id = m.getUniqueID() + "|" + b.getUniqueID();
                            int pred = affPred.containsKey(id) && affPred.get(id) > 0.5 ? 1 : 0;
                            affScores.increment(gold, pred);
                        }
                    }
                }
                affScores.printCompleteScores();;

            }
        } else if(argList.contains("Data")) {
            String featsFileToConvert = parser.getString("convert_to_arff");
            String featsToExtract = parser.getString("extractFeats");
            String neuralPreproc = parser.getString("neuralPreproc");
            String ccaPreproc = parser.getString("ccaPreproc");
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
            } else if(neuralPreproc != null){
                switch(neuralPreproc){
                    case "caption": Preprocess.export_neuralCaptionFile(docSet, _outroot);
                        break;
                    case "relation": Preprocess.export_neuralRelationFiles(docSet, _outroot);
                        break;
                    case "nonvis": Preprocess.export_neuralNonvisFile(docSet, _outroot);
                        break;
                    case "card": Preprocess.export_neuralCardinalityFile(docSet, _outroot);
                        break;
                    case "affinity": Preprocess.export_neuralAffinityFiles(docSet, _outroot);
                        break;
                }
            } else if(ccaPreproc != null){
                switch(ccaPreproc){
                    case "cca_lists": Preprocess.export_phraseLocalization_ccaLists(docSet, split,
                            parser.getString("boxFeatDir"), _outroot);
                        break;
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
            //Set up the relation inference module
            ILPInference.InferenceType infType =
                    ILPInference.InferenceType.valueOf(parser.getString("inf_type").toUpperCase());
            ILPInference inf = new ILPInference(docSet, infType,
                    parser.getString("nonvis_scores"),
                    parser.getString("relation_scores"),
                    parser.getString("affinity_scores"),
                    parser.getString("cardinality_scores"),
                    parser.getString("graph_root"),
                    parser.getBoolean("include_type_constraint"),
                    parser.getBoolean("exclude_box_exigence"),
                    parser.getBoolean("exclude_subset"),
                    parser.getBoolean("only_keep_positive_links"));

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
