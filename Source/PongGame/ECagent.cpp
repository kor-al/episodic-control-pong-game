// Fill out your copyright notice in the Description page of Project Settings.

#include "PongGame.h"
#include "ECagent.h"

const float ECagent::kEpsMax = 1;
const float ECagent::kEpsMin = 0.005;
const float ECagent::kECdiscount = 0.99;
const std::string ECagent::kQECtableFilename = "archive.dat";
//const std::string ECagent::kQECtableFilename = "archive.xml";
const std::string ECagent::kSummaryFilename = "summary.txt";
const std::string ECagent::kKDtreeFilename = "kdtree_index";

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

int ECagent::start_episode(cv::Mat observation, cv::Mat state)
{
	if (!history.history.empty())
		history.history.clear();
	if (!qectable)
		qectable = new QECtable(kKnn, kDimState, observation.rows * observation.cols, kBufferMaxsize, kNumActions, bEmbedding);
	UE_LOG(LogTemp, Warning, TEXT("---start episode"));
	step_counter = 0;
	episode_reward = 0;
	start_time = time(0);
	int action = FMath::RandRange(0, kNumActions);
	last_action = action;
	last_observation = observation;

	if (bEmbedding)
		last_state = state;

	return action;
}

int ECagent::step(cv::Mat observation, int reward, cv::Mat state)
{
	step_counter++;
	episode_reward += reward;
	int action = choose_action(observation, reward, state);
	eps = std::max(kEpsMin, eps - eps_rate);
	last_action = action;
	last_observation = observation;

	if (bEmbedding)
		last_state = state;

	return action;
}

int ECagent::random_action()
{
	return FMath::RandRange(0, kNumActions);
}

int ECagent::choose_action(cv::Mat observation, int reward, cv::Mat state)
{

	if (!bEmbedding)
	{
		history.add_node(last_observation, last_action, reward, false);
		float rand_num = FMath::RandRange(float(0), float(1.0));
		//rand_num = -1; //only rand actions
		if (rand_num<eps)
		{
				int action = random_action();
				UE_LOG(LogTemp, Warning, TEXT("---random step, eps = %f, action = %u, r = %f"), eps, action, rand_num);
				return action;
		}
		float value = -FLT_MAX;
		float value_temp = 0;
		int max_action = 0;
		UE_LOG(LogTemp, Warning, TEXT("---Agent: choose action"));
		//a* = argmaxQ(s,a)
		for (int a = 0; a < kNumActions + 1; a++)
		{
				value_temp = qectable->estimate(observation, a);
				if (value_temp > value)
				{
					UE_LOG(LogTemp, Warning, TEXT("-----value = %f, action = %i"), value_temp, a);
					value = value_temp;
					max_action = a;
				}
		}
		UE_LOG(LogTemp, Warning, TEXT("-------max_value = %f, action = %i"), value, max_action);
		return max_action;
	}
	else
	{

		history.add_node(last_observation, last_action, reward, false, last_state);
		UE_LOG(LogTemp, Warning, TEXT("last state r = %u, c = %u"), last_state.rows, last_state.cols);

		float rand_num = FMath::RandRange(float(0), float(1.0));
		if (rand_num<eps)
		{
			int action = random_action();
			UE_LOG(LogTemp, Warning, TEXT("---random step, eps = %f, action = %u, r = %f"), eps, action, rand_num);
			return action;
		}
		float value = -FLT_MAX;
		float value_temp = 0;
		int max_action = 0;
		UE_LOG(LogTemp, Warning, TEXT("---Agent: choose action"));
		//a* = argmaxQ(s,a)
		for (int a = 0; a < kNumActions + 1; a++)
		{
			value_temp = qectable->estimate(observation, a, state);
			if (value_temp > value)
			{
				UE_LOG(LogTemp, Warning, TEXT("-----value = %f, action = %i"), value_temp, a);
				value = value_temp;
				max_action = a;
			}
		}
		UE_LOG(LogTemp, Warning, TEXT("-------max_value = %f, action = %i"), value, max_action);
		return max_action;
	}	
}


void ECagent::end_episode(int reward, bool bTerminal)
{
	UE_LOG(LogTemp, Warning, TEXT("---end episode"));
	step_counter++;
	episode_reward += reward;
	total_episodes++;
	total_reward += episode_reward;
	total_time = time(0) - start_time;

	if (!bEmbedding)
	{
		history.add_node(last_observation, last_action, reward, true);

		//backup
		float q_t = 0;
		for (int i = history.history.size() - 1; i >= 0; i--)
		{
			HistoryNode node = history.history[i];
			q_t = q_t * kECdiscount + node.reward;
			qectable->update(node.observation, node.action, q_t);
		}
		UE_LOG(LogTemp, Warning, TEXT("-----q return = %f"), q_t, history.history.size());

		if (!(total_episodes % kSaveEpisodes))
		{
			saveQECtable();
			UE_LOG(LogTemp, Warning, TEXT("_____QEC table SAVED_____"));
		}
	}
	else
	{
		history.add_node(last_observation, last_action, reward, true, last_state);

		//backup
		float q_t = 0;
		for (int i = history.history.size() - 1; i >= 0; i--)
		{
			HistoryNode node = history.history[i];
			q_t = q_t * kECdiscount + node.reward;
			//here obs are states
			UE_LOG(LogTemp, Warning, TEXT("end_ep state r = %u, c = %u"), node.observation.rows, node.observation.cols);
			qectable->update(node.observation, node.action, q_t, node.state);
		}
		UE_LOG(LogTemp, Warning, TEXT("-----q return = %f"), q_t, history.history.size());

		if (!(total_episodes % kSaveEpisodes))
		{
			saveQECtable();
			UE_LOG(LogTemp, Warning, TEXT("_____QEC table SAVED_____"));
		}
	}
}


void ECagent::saveQECtable()
{
	std::ofstream file_summary;
	file_summary.open(kSummaryFilename);
	file_summary << "Total Episodes = " << total_episodes << '\n';
	file_summary << "Total Reward = " << total_reward << '\n';
	file_summary.close();


	//std::ofstream file(kQECtableFilename); //for xml achive
	std::ofstream file(kQECtableFilename, std::ios::out | std::ios::binary);
	//boost::archive::xml_oarchive oa(file);
	//oa & BOOST_SERIALIZATION_NVP(qectable);
	boost::archive::binary_oarchive oa(file);
	qectable->eps = eps;
	oa << qectable;
	qectable->save_kdtrees(kKDtreeFilename);
	file.close();
}

void ECagent::loadQECtable()
{
	//std::ifstream file(kQECtableFilename); //for xml achive
	std::ifstream file(kQECtableFilename, std::ios::in | std::ios::binary);
	if (file.good())
	{
		UE_LOG(LogTemp, Warning, TEXT("_____QEC table LOADED_____"));
		//boost::archive::xml_iarchive ia(file);
		//ia >> BOOST_SERIALIZATION_NVP(qectable);
		
		boost::archive::binary_iarchive ia(file);
		ia >> qectable;
		//qectable->load_kdtrees(kKDtreeFilename);
		eps = qectable->eps;
	}
}