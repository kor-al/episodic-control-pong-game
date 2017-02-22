// Fill out your copyright notice in the Description page of Project Settings.

#include "PongGame.h"
#include "ECagent.h"

const float ECagent::kEpsMax = 1;
const float ECagent::kEpsMin = 0.01;
const float ECagent::kECdiscount = 0.99;
const std::string ECagent::kQECtableFilename = "archive.xml";
const std::string ECagent::kSummaryFilename = "summary.txt";
const std::string ECagent::kKDtreeFilename = "kdtree_index.fln";

ECagent::ECagent()
{

	total_reward = 0;
	total_episodes = 0;
	qectable = 0;
	eps = kEpsMax;

	if (kEpsDenominator > 0)
	{
		eps_rate = (kEpsMax - kEpsMin) / kEpsDenominator;
	}
	else
		eps_rate = 0;
}

ECagent::~ECagent()
{
	delete qectable;
}

int ECagent::start_episode(cv::Mat observation)
{
	if (!history.history.empty())
		history.history.clear();
	if (!qectable)
		qectable = new QECtable(kKnn, kDimState, observation.rows * observation.cols, kBufferMaxsize, kNumActions);
	UE_LOG(LogTemp, Warning, TEXT("start episode"));
	step_counter = 0;
	episode_reward = 0;
	start_time = time(0);
	int action = FMath::RandRange(0, kNumActions);
	last_action = action;
	last_observation = observation;
	return action;
}

int ECagent::step(cv::Mat observation, int reward)
{
	step_counter++;
	episode_reward += reward;
	int action = choose_action(observation, reward);
	eps = std::max(kEpsMin, eps - eps_rate);
	last_action = action;
	last_observation = observation;
	return action;
}

int ECagent::choose_action(cv::Mat observation, int reward)
{
	history.add_node(last_observation, last_action, reward, false);
	float rand_num = FMath::RandRange(float(0), float(1.0));
	if (rand_num<eps)
	{
		int action = FMath::RandRange(0, kNumActions);
		//UE_LOG(LogTemp, Warning, TEXT("random step, action = %d, r = %f"), action, rand_num);
		return action;
	}
	double value = -DBL_MAX;
	double value_temp = 0;
	int max_action = 0;
	//a* = argmaxQ(s,a)
	for (int a = 0; a < kNumActions + 1; a++)
	{
		value_temp = qectable->estimate(observation, a);
		if (value_temp > value)
		{
			value = value_temp;
			max_action = a;
		}

	}
	return max_action;
}


void ECagent::end_episode(int reward, bool bTerminal)
{
	UE_LOG(LogTemp, Warning, TEXT("end episode"));
	step_counter++;
	episode_reward += reward;
	total_episodes++;
	total_reward += episode_reward;
	total_time = time(0) - start_time;
	history.add_node(last_observation, last_action, reward, true);

	//backup
	double q_t = 0;
	for (int i = history.history.size() - 1; i >= 0; i--)
	{
		HistoryNode node = history.history[i];
		q_t = q_t * kECdiscount + node.reward;
		qectable->update(node.observation, node.action, q_t);
	}
	UE_LOG(LogTemp, Warning, TEXT("q return = %f"), q_t, history.history.size());

	if (!(total_episodes % kSaveEpisodes))
	{
		saveQECtable();
	}
}

//ECagent(const ECagent& that)
//{
//	qectable = new QECtable(kKNN, that.qectable.matrix_proj.rows, that.qectable.matrix_proj.cols, kBufferMaxsize, kNumActions);
//	step_counter = that.step_counter;
//	episode_reward = that.episode_reward;
//	total_reward = that.total_reward;
//	total_episodes = that.total_episodes;
//	start_time = that.start_time;
//	total_time = that.total_time;
//	last_observation = that.last_observation;
//	last_action = that.last_action;
//	history = that.history;
//}
//ECagent& operator=(const ECagent& that)
//{
//
//	return *this;
//}

//void ECagent::finish_play();

void ECagent::saveQECtable()
{
	std::ofstream file_summary;
	file_summary.open(kSummaryFilename);
	file_summary << "Total Episodes = " << total_episodes << '\n';
	file_summary << "Total Reward = " << total_reward << '\n';
	file_summary.close();

	std::ofstream file(kQECtableFilename);
	boost::archive::xml_oarchive oa(file);
	//boost::archive::binary_oarchive oa(file);
	oa & BOOST_SERIALIZATION_NVP(qectable);
	qectable->save_kdtrees(kKDtreeFilename);
	file.close();
}

void ECagent::loadQECtable()
{
	std::ifstream file(kQECtableFilename);
	if (file.good())
	{
		UE_LOG(LogTemp, Warning, TEXT("QEC table LOADED"));
		boost::archive::xml_iarchive ia(file);
		//boost::archive::binary_iarchive ia(file);
		ia >> BOOST_SERIALIZATION_NVP(qectable);
		qectable->load_kdtrees(kKDtreeFilename);
	}
}