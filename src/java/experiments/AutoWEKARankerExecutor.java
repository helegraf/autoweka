package experiments;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import de.upb.crc901.mlplan.core.events.ClassifierFoundEvent;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import experiments.two_part.part_two.execution.RankerConfig;
import experiments.two_part.part_two.execution.RankerExecutor;
import jaicore.basic.SQLAdapter;
import jaicore.ml.WekaUtil;
import ranker.core.algorithms.Ranker;
import weka.core.Instances;

public class AutoWEKARankerExecutor extends RankerExecutor {

	private Logger logger = LoggerFactory.getLogger(AutoWEKARankerExecutor.class);
	private SQLAdapter adapter;
	private String intermediateResultsTable;
	private int experimentId;
	private AutoWEKARankerConfig configuration;

	@Override
	protected String getActiveConfiguration() {
		Properties properties = new Properties();
		properties.setProperty("autoweka.seed", String.valueOf(configuration.getSeed()));
		properties.setProperty("autoweka.memory", String.valueOf(configuration.getMemory()));
		properties.setProperty("autoweka.numCPUs", String.valueOf(configuration.getNumCPUs()));
		properties.setProperty("autoweka.totalTimeoutSeconds", String.valueOf(configuration.getTotalTimeoutSeconds()));
		properties.setProperty("autoweka.evaluationTimeoutSeconds", String.valueOf(configuration.getEvaluationTimeoutSeconds()));
		return properties.toString();
	}

	@Override
	protected Ranker getOptimalRanker(Instances hyperTrain, Instances hyperTest, List<Integer> targetAttributes,
			RankerConfig configuration) {
		return this.instantiate(configuration);
	}

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return AutoWEKARankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig config) {
		configuration = (AutoWEKARankerConfig) config;

		try {
			if (configuration.uploadIntermediateResults()) {
				adapter = new SQLAdapter(configuration.getHost(), configuration.getUser(), configuration.getPassword(),
						configuration.getDatabase());
				intermediateResultsTable = configuration.getIntermediateResultsTable();
				experimentId = super.getExperimentId();
				createIntermediateResultsTableIfNotExists(intermediateResultsTable);
			}
		} catch (SQLException e) {
			logger.warn("Will not upload intermediate results due to {}", e);
			adapter = null;
		}

		return new AutoWEKARegressionRanker(configuration.getMemory(), this.getExperimentId(),
				configuration.getNumCPUs(), configuration.getTotalTimeoutSeconds(),
				configuration.getEvaluationTimeoutSeconds());
	}

	private void createIntermediateResultsTableIfNotExists(String table) throws SQLException {
		String sql = String.format("CREATE TABLE IF NOT EXISTS `%s` (\r\n"
				+ " `evaluation_id` int(10) NOT NULL AUTO_INCREMENT,\r\n" + " `experiment_id` int(8) NOT NULL,\r\n"
				+ " `preprocessor` text COLLATE utf8_bin NOT NULL,\r\n"
				+ " `classifier` text COLLATE utf8_bin NOT NULL,\r\n" + " `rmse` double DEFAULT NULL,\r\n"
				+ " `time_train` bigint(8) DEFAULT NULL,\r\n" + " `time_predict` int(8) DEFAULT NULL,\r\n"
				+ " `evaluation_timestamp_finish` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,\r\n"
				+ " `exception` text COLLATE utf8_bin,\r\n" + " PRIMARY KEY (`evaluation_id`)\r\n"
				+ ") ENGINE=InnoDB AUTO_INCREMENT=8281430 DEFAULT CHARSET=utf8 COLLATE=utf8_bin", table);

		adapter.update(sql);
	}

	@Subscribe
	public void rcvHASCOSolutionEvent(final ClassifierFoundEvent e) {
		if (adapter != null) {
			try {
				String classifier = "";
				String preprocessor = "";
				if (e.getSolutionCandidate() instanceof MLPipeline) {
					MLPipeline solution = (MLPipeline) e.getSolutionCandidate();
					preprocessor = solution.getPreprocessors().isEmpty() ? ""
							: solution.getPreprocessors().get(0).toString();
					classifier = WekaUtil.getClassifierDescriptor(solution.getBaseClassifier());
				} else {
					classifier = WekaUtil.getClassifierDescriptor(e.getSolutionCandidate());
				}
				Map<String, Object> eval = new HashMap<>();
				eval.put("experiment_id", experimentId);
				eval.put("preprocessor", preprocessor);
				eval.put("classifier", classifier);
				if (!Double.isNaN(e.getScore())) {
					eval.put("rmse", e.getScore());
				} else {
					logger.warn("Uploading incomplete intermediate solution!");
				}

				eval.put("time_train", e.getTimestamp());
				adapter.insert(intermediateResultsTable, eval);
			} catch (Exception e1) {
				logger.error("Could not store hasco solution in database", e1);
			}
		} else {
			logger.error("no adapter!");
		}
	}

}
