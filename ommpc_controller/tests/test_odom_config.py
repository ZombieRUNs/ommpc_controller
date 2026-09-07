"""Check odometry defaults and the launch file's resolved private parameters."""
import pathlib
import unittest

import roslaunch.config
import yaml


PACKAGE = pathlib.Path(__file__).resolve().parents[1]


class OdomConfigTest(unittest.TestCase):
    def test_yaml_defaults(self):
        params = yaml.safe_load((PACKAGE / "config/params.yaml").read_text())
        self.assertEqual(params["odom_topic"], "/mavros/local_position/odom")
        self.assertIs(params["vel_in_body"], True)

    def test_launch_loads_private_parameters(self):
        config = roslaunch.config.load_config_default(
            [str(PACKAGE / "launch/px4_example.launch")], None
        )
        self.assertEqual(
            config.params["/ommpc_controller/odom_topic"].value,
            "/mavros/local_position/odom",
        )
        self.assertIs(config.params["/ommpc_controller/vel_in_body"].value, True)

    def test_cpp_reads_private_parameters_and_subscribes(self):
        source = (PACKAGE / "src/ommpc_example.cpp").read_text()
        self.assertIn('ros::NodeHandle nh("~");', source)
        self.assertIn('ommpc_example.init(nh);', source)
        self.assertIn('nh.param("vel_in_body", vel_in_body_, true);', source)
        self.assertIn(
            'nh.param<std::string>("odom_topic", odom_topic, '
            '"/mavros/local_position/odom");', source
        )
        self.assertIn('nh.subscribe<nav_msgs::Odometry>(odom_topic,', source)
        self.assertIn('enu_frame_ = true;', source)
        self.assertIn('odom_data_.feed(msg, enu_frame_, vel_in_body_);', source)


if __name__ == "__main__":
    unittest.main()
