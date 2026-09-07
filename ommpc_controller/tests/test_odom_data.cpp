#include <gtest/gtest.h>
#include "ommpc_controller.hpp"

namespace
{
void checkEnuOdometry(const Eigen::Quaterniond &attitude,
                      const Eigen::Vector3d &world_velocity, bool vel_in_body)
{
  ros::Time::init();
  nav_msgs::OdometryPtr msg(new nav_msgs::Odometry);
  msg->pose.pose.position.x = 4.0;
  msg->pose.pose.position.y = -5.0;
  msg->pose.pose.position.z = 6.0;
  msg->pose.pose.orientation.w = attitude.w();
  msg->pose.pose.orientation.x = attitude.x();
  msg->pose.pose.orientation.y = attitude.y();
  msg->pose.pose.orientation.z = attitude.z();
  msg->twist.twist.linear.x = 1.0;
  msg->twist.twist.linear.y = 2.0;
  msg->twist.twist.linear.z = 3.0;
  msg->twist.twist.angular.x = 0.1;
  msg->twist.twist.angular.y = -0.2;
  msg->twist.twist.angular.z = 0.3;

  Odom_Data_t odom;
  odom.feed(msg, true, vel_in_body);

  EXPECT_TRUE(odom.p.isApprox(Eigen::Vector3d(4.0, -5.0, 6.0), 1e-12));
  EXPECT_TRUE(odom.q.coeffs().isApprox(attitude.coeffs(), 1e-12));
  EXPECT_TRUE(odom.w.isApprox(Eigen::Vector3d(0.1, -0.2, 0.3), 1e-12));
  EXPECT_TRUE(odom.v.isApprox(world_velocity, 1e-12));
  EXPECT_TRUE(odom.recv_new_msg);
}
}

TEST(OdomData, BodyFluVelocityRotatesToEnuWithYawAndTilt)
{
  const Eigen::Quaterniond yaw(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
  const Eigen::Quaterniond pitch(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond roll(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()));
  checkEnuOdometry(yaw, Eigen::Vector3d(-2.0, 1.0, 3.0), true);
  checkEnuOdometry(yaw * pitch, Eigen::Vector3d(-2.0, 3.0, -1.0), true);
  checkEnuOdometry(yaw * roll, Eigen::Vector3d(3.0, 1.0, 2.0), true);
}

TEST(OdomData, WorldEnuVelocityIsNotRotatedWithYawAndTilt)
{
  const Eigen::Quaterniond yaw(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
  const Eigen::Quaterniond pitch(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond roll(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()));
  const Eigen::Vector3d velocity(1.0, 2.0, 3.0);
  checkEnuOdometry(yaw, velocity, false);
  checkEnuOdometry(yaw * pitch, velocity, false);
  checkEnuOdometry(yaw * roll, velocity, false);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
