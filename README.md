# roma_cnn_count

This repo is an extension of the **[Confidence S-COUNT](https://github.com/CSCarbone07/roma_confident_scount_multi/blob/master/README.md)** to 
be included in ROS for the AgroSwarm simulator.

Check out more in the documentation at

**[AgroSwarm Documentation](https://roma-agroswarm-quadai.readthedocs.io/en/latest/home.html)**


Default network weights for detection of target on cardboard boxes can be downloaded at

**[Cardboard box target detection weights](https://034e925c-962c-4fbe-8a23-5266f414f783.usrfiles.com/archives/034e92_2504b5e55d64400abc878c7e3f82bed6.zip)**

To use the default weights make sure the downloaded pth file is included within the path (assuming your ROS workspace is at ~/catkin_ws_rm):

~/catkin_ws_rm/src/roma_cnn_count/test.pth
