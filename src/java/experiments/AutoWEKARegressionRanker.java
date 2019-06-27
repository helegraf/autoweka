package experiments;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.ml.WekaUtil;
import ranker.core.algorithms.decomposition.DecompositionRanker;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AutoWEKAClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class AutoWEKARegressionRanker extends DecompositionRanker {

	private int seed;
	private int numCPUS;
	private int totalTimeoutSeconds;
	private int evaluationTimeoutSeconds;;
	private int memory;

	private static final Logger LOGGER = LoggerFactory.getLogger(AutoWEKARegressionRanker.class);

	public AutoWEKARegressionRanker(int memory, int seed, int numCPUS, int totalTimeoutSeconds,
			int evaluationTimeoutSeconds) {
		this.seed = seed;
		this.numCPUS = numCPUS;
		this.totalTimeoutSeconds = totalTimeoutSeconds;
		this.evaluationTimeoutSeconds = evaluationTimeoutSeconds;
		this.memory = memory;
	}

	@Override
	protected void buildModels(Map<String, Instances> train) throws Exception {
		models = new HashMap<>();

		train.forEach((classifier, instances) -> {

			/*
			 * initialize ML-Plan with the same config file that has been used to specify
			 * the experiments
			 */
			try {
				// experimentEntry.getExperiment().getMemoryInMB()
				AutoWEKAClassifier autoweka = new AutoWEKAClassifier();
				autoweka.setSeed(seed);
				autoweka.setMemLimit(memory);
				autoweka.setMetric(AutoWEKAClassifier.Metric.rootMeanSquaredError);
				autoweka.setTimeLimit((totalTimeoutSeconds / 60) / train.size());
				autoweka.setSingleAlgorithmTimeLimitSeconds(evaluationTimeoutSeconds);
				// autoweka.setSingleAlgorithmTimeLimitSeconds(5);
				autoweka.setParallelRuns(numCPUS);

				LOGGER.info("Build autoweka classifier");
				autoweka.buildClassifier(instances);

				models.put(classifier, autoweka);
			} catch (Exception e) {
				LOGGER.warn("Could not train model for {} due to {}, using random forest.",
						classifier, e);

				RandomForest forest = new RandomForest();
				try {
					forest.buildClassifier(instances);
				} catch (Exception e1) {
					LOGGER.error("Could not train model for {} due to {}, using random forest.",
							classifier, e);
				}
				models.put(classifier, forest);
			}

		});
	}

	public String getSelectedModelString() {
		StringBuilder builder = new StringBuilder();

		this.models.values().forEach(autoweka -> {
			if (autoweka instanceof AutoWEKAClassifier) {
				AutoWEKAClassifier classif = (AutoWEKAClassifier) autoweka;
				Classifier selectedClassifier = classif.getClassifier();
				String attributeEvalClass = classif.getAttributeEvalClass();
				String attributeEvalArgs = classif.getAttributeEvalArgs() == null ? ""
						: Arrays.toString(classif.getAttributeEvalArgs());
				String attributeSearchClass = classif.getAttributeSearchClass() == null ? ""
						: classif.getAttributeSearchClass();
				String attributeSearchArgs = classif.getAttributeSearchArgs() == null ? ""
						: Arrays.toString(classif.getAttributeSearchArgs());

				if (attributeEvalClass != null) {
					builder.append("pl: [");
					builder.append("pre: [");
					builder.append(attributeEvalClass + " " + attributeEvalArgs + " " + attributeSearchClass + " "
							+ attributeSearchArgs);
					builder.append("] class: [");
					builder.append(classif.getClassifier().getClass().getName() + " "
							+ Arrays.toString(classif.getClassifierArgs()));
					builder.append("], ");
				} else {
					builder.append("class: [");
					builder.append(WekaUtil.getClassifierDescriptor(selectedClassifier));
					builder.append("], ");
				}
			} else {
				builder.append("class: [");
				builder.append(WekaUtil.getClassifierDescriptor(autoweka));
				builder.append("], ");
			}

		});

		return builder.toString();
	}

	@Override
	public String getName() {
		return super.getName() + "_" + seed + "_" + numCPUS + "_" + totalTimeoutSeconds + "_" + evaluationTimeoutSeconds
				+ "_" + memory;
	}

	@Override
	public String getClassifierString() {
		return getSelectedModelString();
	}

}
