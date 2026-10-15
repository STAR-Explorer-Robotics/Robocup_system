#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

namespace super_odometry {
namespace {

Eigen::Isometry3d makeTransform(const std::vector<double> &xyzrpy, bool rpy_degrees) {
    const double scale = rpy_degrees ? M_PI / 180.0 : 1.0;
    const double roll = xyzrpy[3] * scale;
    const double pitch = xyzrpy[4] * scale;
    const double yaw = xyzrpy[5] * scale;

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(xyzrpy[0], xyzrpy[1], xyzrpy[2]);
    transform.linear() =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    return transform;
}

Eigen::Vector3d vectorFromMsg(const geometry_msgs::msg::Vector3 &msg) {
    return Eigen::Vector3d(msg.x, msg.y, msg.z);
}

void vectorToMsg(const Eigen::Vector3d &value, geometry_msgs::msg::Vector3 &msg) {
    msg.x = value.x();
    msg.y = value.y();
    msg.z = value.z();
}

}  // namespace

class OdomFrameTransformNode : public rclcpp::Node {
public:
    OdomFrameTransformNode() : Node("odom_frame_transform_node") {
        input_topic_ = declare_parameter<std::string>("input_odom_topic", "/imu_odometry");
        output_topic_ = declare_parameter<std::string>("output_odom_topic", "/body_odometry");
        output_child_frame_ = declare_parameter<std::string>("output_child_frame", "base_link");
        publish_tf_ = declare_parameter<bool>("publish_tf", true);
        const bool rpy_degrees = declare_parameter<bool>("extrinsic_rpy_degrees", false);
        const auto xyzrpy = readXyzRpyParameter("lidar_to_body_xyzrpy");
        lidar_to_body_ = makeTransform(xyzrpy, rpy_degrees);
        body_from_lidar_rotation_ = lidar_to_body_.linear();
        lidar_to_body_inverse_ = lidar_to_body_.inverse();

        pub_ = create_publisher<nav_msgs::msg::Odometry>(output_topic_, 20);
        sub_ = create_subscription<nav_msgs::msg::Odometry>(
            input_topic_, 50,
            std::bind(&OdomFrameTransformNode::odomHandler, this, std::placeholders::_1));
        if (publish_tf_) {
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }

        RCLCPP_INFO(get_logger(),
                    "Odometry frame transform: %s -> %s, output child_frame=%s, publish_tf=%s",
                    input_topic_.c_str(), output_topic_.c_str(), output_child_frame_.c_str(),
                    publish_tf_ ? "true" : "false");
    }

private:
    std::vector<double> readXyzRpyParameter(const std::string &name) {
        auto value = declare_parameter<std::vector<double>>(
            name, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        if (value.size() != 6) {
            RCLCPP_WARN(get_logger(),
                        "%s must contain [x, y, z, roll, pitch, yaw]. Falling back to identity.",
                        name.c_str());
            value = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        return value;
    }

    void odomHandler(const nav_msgs::msg::Odometry::SharedPtr msg) {
        Eigen::Quaterniond q_world_lidar(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z);
        if (q_world_lidar.norm() < 1e-9) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Dropping odometry with invalid orientation norm.");
            return;
        }
        q_world_lidar.normalize();

        Eigen::Isometry3d world_to_lidar = Eigen::Isometry3d::Identity();
        world_to_lidar.linear() = q_world_lidar.toRotationMatrix();
        world_to_lidar.translation() = Eigen::Vector3d(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z);

        const Eigen::Isometry3d world_to_body = world_to_lidar * lidar_to_body_inverse_;
        Eigen::Quaterniond q_world_body(world_to_body.linear());
        q_world_body.normalize();

        nav_msgs::msg::Odometry output = *msg;
        output.child_frame_id = output_child_frame_;
        output.pose.pose.position.x = world_to_body.translation().x();
        output.pose.pose.position.y = world_to_body.translation().y();
        output.pose.pose.position.z = world_to_body.translation().z();
        output.pose.pose.orientation.x = q_world_body.x();
        output.pose.pose.orientation.y = q_world_body.y();
        output.pose.pose.orientation.z = q_world_body.z();
        output.pose.pose.orientation.w = q_world_body.w();

        vectorToMsg(body_from_lidar_rotation_ * vectorFromMsg(msg->twist.twist.linear),
                    output.twist.twist.linear);
        vectorToMsg(body_from_lidar_rotation_ * vectorFromMsg(msg->twist.twist.angular),
                    output.twist.twist.angular);

        pub_->publish(output);
        if (tf_broadcaster_) {
            publishTransform(output);
        }
    }

    void publishTransform(const nav_msgs::msg::Odometry &odom) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header = odom.header;
        transform.child_frame_id = odom.child_frame_id;
        transform.transform.translation.x = odom.pose.pose.position.x;
        transform.transform.translation.y = odom.pose.pose.position.y;
        transform.transform.translation.z = odom.pose.pose.position.z;
        transform.transform.rotation = odom.pose.pose.orientation;
        tf_broadcaster_->sendTransform(transform);
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string output_child_frame_;
    bool publish_tf_;
    Eigen::Isometry3d lidar_to_body_ = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d lidar_to_body_inverse_ = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d body_from_lidar_rotation_ = Eigen::Matrix3d::Identity();

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace super_odometry

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<super_odometry::OdomFrameTransformNode>());
    rclcpp::shutdown();
    return 0;
}
