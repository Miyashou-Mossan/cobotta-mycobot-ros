#include "ros/ros.h"
#include "sensor_msgs/JointState.h"
#include "trajectory_msgs/JointTrajectory.h"
#include <vector>
#include <iostream>
#include <mutex>

// --- グローバル変数と定数 ---
const std::vector<std::string> JOINT_NAMES = {
    "joint_1", 
    "joint_2", 
    "joint_3", 
    "joint_4", 
    "joint_5", 
    "joint_6"
};

// 【新規】現在のジョイント位置を保持するグローバル変数
// 初期値はURDFとMoveIt!のデフォルト姿勢（通常は0.0）と合わせる
std::vector<double> current_joint_positions = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
std::mutex state_mutex; 


// --- サブスクライバーのコールバック関数 (修正箇所) ---
void trajectoryCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg)
{
    ROS_INFO("--- Received Joint Trajectory Command ---");
    
    // データ更新のため、ロックで保護
    std::lock_guard<std::mutex> lock(state_mutex); 

    if (!msg->points.empty()) {
        
        // 1. ジョイント名と軌道全体の情報を表示
        ROS_INFO("Joint Names (%zu):", msg->joint_names.size());
        for (const auto& name : msg->joint_names) {
            std::cout << "  - " << name << "\n";
        }
        
        ROS_INFO("Trajectory received with %zu points:", msg->points.size());
        
        // 2. すべてのポイントをループ処理して詳細を表示
        for (size_t p_index = 0; p_index < msg->points.size(); ++p_index) {
            const auto& point = msg->points[p_index];
            
            // ポイント番号と時刻を表示
            std::cout << "  [POINT " << p_index << "] Time: " << point.time_from_start.toSec() << "s\n";
            
            // 位置データを整形して表示
            std::cout << "    Positions: [";
            for (size_t j_index = 0; j_index < point.positions.size(); ++j_index) {
                // 各ジョイントの位置を小数点以下4桁まで出力
                std::cout << std::fixed << std::setprecision(4) << point.positions[j_index] 
                          << (j_index == point.positions.size() - 1 ? "" : ", ");
            }
            std::cout << "]\n";
        }
        
        // 3. 【瞬間移動ロジック】: 最終目標姿勢を現在の状態として保存
        const auto& last_point = msg->points.back();
        if (last_point.positions.size() == JOINT_NAMES.size())
        {
            current_joint_positions = last_point.positions;
            ROS_INFO("State updated to final goal position. (Model will snap in Rviz)");
        }
        else
        {
            ROS_WARN("Joint position size mismatch in trajectory command.");
        }

    } else {
        ROS_WARN("Received JointTrajectory message with no points.");
    }
    ROS_INFO("-----------------------------------------");
}


// --- メイン関数 (修正箇所) ---
int main(int argc, char **argv)
{
    // ROSノードの初期化...
    ros::init(argc, argv, "dummy_cobotta_ros1_driver");
    ros::NodeHandle nh;

    // 1. JointTrajectory サブスクライバーの設定 (変更なし)
    ros::Subscriber sub = nh.subscribe(
        "/cobotta/joint_command",
        1, 
        trajectoryCallback
    );

    // 2. JointState パブリッシャーの設定 (変更なし)
    ros::Publisher pub = nh.advertise<sensor_msgs::JointState>(
        "/cobotta/state",
        1
    );

    // 3. パブリッシュメッセージの準備
    sensor_msgs::JointState joint_state_msg;
    joint_state_msg.name = JOINT_NAMES;
    joint_state_msg.velocity.assign(JOINT_NAMES.size(), 0.0);
    
    // パブリッシュレートの設定
    ros::Rate loop_rate(100); 

    ROS_INFO("Dummy ROS 1 Driver initialized. Publishing JointState at 100Hz and listening for JointTrajectory.");

    // メインループ (修正箇所)
    while (ros::ok())
    {
        // 【修正点】: 現在のジョイント位置をパブリッシュメッセージに設定
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            joint_state_msg.position = current_joint_positions; 
        }

        joint_state_msg.header.stamp = ros::Time::now();
        pub.publish(joint_state_msg);

        // コールバック処理とスリープ
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}