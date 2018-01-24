package core;

import learn.ClassifyUtil;
import learn.ILPInference;
import learn.Preprocess;
import structures.Document;
import structures.Mention;
import utilities.*;

import java.util.*;

import static core.DocumentLoader.getDocumentSet;

/**The Main is responsible for
 * parsing args and calling other ImageCaptionLearn
 * modules
 * 
 * @author ccervantes
 */
public class Main
{
    //Paths
    public static String dataPath, captionTePath, wordnetDir,
            word2vecPath, boxFeatureDir, flickr30kPath,
            flickr30kPath_v1, flickr30k_sqlite, flickr30k_sqlite_v1,
            flickr30kResources, flickr30k_lexicon, mscocoPath,
            mscoco_sqlite, mscocoResources, mscoco_lexicon,
            snliPath, denotation_sqlite, mpe_sqlite;
    public static String[] flickr30k_mysqlParams, mscoco_mysqlParams;
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

        //Load config file for paths
        Map<String, String> configDict = new HashMap<>();
        for(String line : FileIO.readFile_lineList("paths.config")){
            if(!line.isEmpty()) {
                String[] kv = line.split("=");
                configDict.put(kv[0].trim(), kv[1].trim());
            }
        }
        dataPath = configDict.get("dataPath");
        captionTePath = configDict.get("captionTePath");
        wordnetDir = configDict.get("wordnetDir");
        word2vecPath = configDict.get("word2vecPath");
        boxFeatureDir = configDict.get("boxFeatureDir");
        flickr30kPath = configDict.get("flickr30kPath");
        flickr30kPath_v1 = configDict.get("flickr30kPath_v1");
        flickr30k_mysqlParams = configDict.get("flickr30k_mysqlParams").split(" ");
        flickr30k_sqlite = configDict.get("flickr30k_sqlite");
        flickr30k_sqlite_v1 = configDict.get("flickr30k_sqlite_v1");
        flickr30kResources = configDict.get("flickr30kResources");
        flickr30k_lexicon = configDict.get("flickr30k_lexicon");
        mscocoPath = configDict.get("mscocoPath");
        mscoco_mysqlParams = configDict.get("mscoco_mysqlParams").split(" ");
        mscoco_sqlite = configDict.get("mscoco_sqlite");
        mscocoResources = configDict.get("mscocoResources");
        mscoco_lexicon = configDict.get("mscoco_lexicon");
        snliPath = configDict.get("snliPath");
        denotation_sqlite = configDict.get("denotation_sqlite");
        mpe_sqlite = configDict.get("mpe_sqlite");

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
        String[] datasetOpts = {"flickr30k", "mscoco", "flickr30k_v1", "mpe"};
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
        /* Data Group */
        parser.addSubParser("Data");
        String[] featOpts = {"relation", "affinity", "nonvis", "card"};
        parser.setArgument_opts("--extractFeats", featOpts, null,
                "Extracts features to --out", "Data");
        parser.setArgument_opts("--neuralPreproc",
                new String[]{"caption", "nonvis", "relation", "card", "affinity", "mpe"},
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
        String[] dbOpts = {"mysql", "sqlite"};
        parser.setArgument_opts("--buildDB", dbOpts, "mysql",
                "Rebuilds the specified database in the default locations", "Data");

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
        List<ILPInference.InferenceType> infTypes = Arrays.asList(ILPInference.InferenceType.values());
        List<String> infTypeStrs = new ArrayList<>();
        infTypes.forEach(t -> infTypeStrs.add(t.toString().toLowerCase()));
        parser.setArgument_opts("--inf_type", infTypeStrs.toArray(new String[]{}), null,
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
            } else if(dataset.equals(datasetOpts[2])){
                conn = new DBConnector(flickr30k_sqlite_v1);
            } else if(dataset.equals(datasetOpts[2]) && runLocal){
                conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1],
                        flickr30k_mysqlParams[2], flickr30k_mysqlParams[3] + "_legacy");
            } else if(dataset.equals(datasetOpts[3])){
                conn = new DBConnector(mpe_sqlite);
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
                Mention.initializeLexicons(Main.flickr30k_lexicon, Main.mscoco_lexicon);
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
            _debug();
        } else if(argList.contains("Data")) {
            String featsToExtract = parser.getString("extractFeats");
            String neuralPreproc = parser.getString("neuralPreproc");
            String ccaPreproc = parser.getString("ccaPreproc");
            String buildDB = parser.getString("buildDB");

            if(featsToExtract != null){
                if(featsToExtract.equals("relation")){
                    ClassifyUtil.exportFeatures_relation(docSet, _outroot, numThreads,
                            parser.getBoolean("for_neural"),
                            !parser.getBoolean("exclude_subset"), false,
                            false, null);
                }
                else if(featsToExtract.equals("affinity")) {
                    ClassifyUtil.exportFeatures_affinity(docSet, split);
                } else if(featsToExtract.equals("nonvis")) {
                    ClassifyUtil.exportFeatures_nonvis(docSet, _outroot,
                            parser.getBoolean("for_neural"),
                            false, null);
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
                    case "mpe": Preprocess.export_neuralMPEFile(docSet, _outroot);
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
                    Preprocess.buildImageCaptionDB(Main.flickr30kPath + "Flickr30kEntities_v2.coref",
                            Main.flickr30kPath + "RELEASE/", Main.flickr30kResources + "img_comments.csv",
                            Main.flickr30kResources + "img_crossval.csv", Main.flickr30kResources + "img_reviewed.txt",
                            flickr30k_mysqlParams[0], flickr30k_mysqlParams[1], flickr30k_mysqlParams[2], flickr30k_mysqlParams[3]);
                } else if(buildDB.equals("sqlite")){
                    Preprocess.buildImageCaptionDB(Main.flickr30kPath + "Flickr30kEntities_v2.coref",
                            Main.flickr30kPath + "RELEASE/", Main.flickr30kResources + "img_comments.csv",
                            Main.flickr30kResources + "img_crossval.csv", Main.flickr30kResources + "img_reviewed.txt",
                            Main.flickr30kPath + "Flickr30kEntities_v2_" +Util.getCurrentDateTime("yyyyMMdd") + ".db");
                }
            }
        } else if(argList.contains("Infer")) {
            Mention.initializeLexicons(Main.flickr30k_lexicon, Main.mscoco_lexicon);

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

    /**Contains as-needed debug code; typically should be empty at checkin
     */
	private static void _debug()
    {

    }
}
