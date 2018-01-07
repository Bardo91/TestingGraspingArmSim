//
//
//
//
//

#include <iostream>
#include <thread>
#include <chrono>

#include <grasping_tools/grasping_tools.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include <arm_controller/visualization/ArmVis.h>
#include <arm_controller/Arm.h>

bool run = true;
bool next = false;

void callback(const pcl::visualization::KeyboardEvent & _event, void * _ptrViewer){
    if (_event.getKeySym() == "Escape" && _event.keyDown()) {
        run = false;
    }else if (_event.getKeySym() == "n" && _event.keyDown()) {
        next = true;
    }
}


int main(int _argc, char**_argv){
    grasping_tools::ObjectMesh object(_argv[1]);
    pcl::PolygonMesh gripperModel;
    bool displayGripper = pcl::io::loadPolygonFileSTL("/home/bardo91/programming/catkin_grasping/src/hecatonquiros/cad_models/tools/gripper/pwm_model/stl/gripper_complete.stl", gripperModel) != 0;
    
    pcl::PolygonMesh mesh;
    arma::mat44 poseObj = arma::eye(4,4);
    poseObj(0,3) = 0.25;
    poseObj(1,3) = 0.25;
    poseObj(2,3) = 0.2;
    poseObj(0,0) = cos(M_PI/4); poseObj(0,1) = -sin(M_PI/4); 
    poseObj(1,0) = sin(M_PI/4); poseObj(1,1) = cos(M_PI/4); 
    object.move(poseObj);
    object.mesh(mesh);

    grasping_tools::GripperHandTwoFingers gripper(0.15);
    pcl::visualization::PCLVisualizer viewer;

    viewer.registerKeyboardCallback(callback);

    Arm arm;
    ArmVis armVis;
    armVis.init(&viewer, "/home/bardo91/programming/catkin_grasping/src/hecatonquiros/cad_models/arm_3_dog/pwm_model/stl");

    while(run){
        viewer.removeAllShapes();
        viewer.removeAllPointClouds();
        viewer.removeAllCoordinateSystems();

        viewer.addCoordinateSystem(0.1);    
        
        auto grasp = gripper.generate(object);

        viewer.addPolygonMesh(mesh, "mesh");

        auto cps = grasp.contactPoints();
        
        if(cps.size() >= 2){
            std::cout << "New grasp" <<std::endl;
            std::cout << "has closure: " << grasp.hasForceClosure() << std::endl;
            std::cout << "lmrw: " << grasp.lmrw() << std::endl;
            for(unsigned i = 0; i < cps.size(); i++){
                grasping_tools::plotContactPoint(cps[i], viewer, 0.1, "cone_"+std::to_string(i));
            }

            arma::mat candidatePoints = grasping_tools::pointsInCircle( 0.2,
                                                                        (cps[0].position() + cps[1].position())/2,
                                                                        cps[0].normal(),
                                                                        100);

            pcl::PointCloud<pcl::PointXYZRGB> candidatePointsPcl;

            int idxMin = -1;
            float minAngleDist = std::numeric_limits<float>::max();
            for(unsigned i = 0; i < candidatePoints.n_cols; i++){
                Eigen::Vector3f targetPosition = {candidatePoints(0,i), candidatePoints(1,i), candidatePoints(2,i)};
                std::vector<Eigen::Matrix4f> tfs;
                std::vector<float> angles;
                if(arm.checkIk(targetPosition, angles, tfs)){
                    pcl::PointXYZRGB p(0,255,0);
                    p.x = candidatePoints(0,i);
                    p.y = candidatePoints(1,i);
                    p.z = candidatePoints(2,i);
                    candidatePointsPcl.push_back(p);

                    Eigen::Matrix4f toolAxis = tfs[0]*tfs[1]*tfs[2];
                    arma::colvec3 n1 = (cps[0].position() + cps[1].position()) / 2 - candidatePoints.col(i);
                    n1 /= arma::norm(n1);

                    arma::colvec3 n2 = { toolAxis(0,0), toolAxis(1,0), toolAxis(2,0) };
                    float dist = acos(arma::dot(n1, n2));
                    if(dist < minAngleDist){
                        minAngleDist = dist;
                        idxMin = i;
                    }
                }
            }
            viewer.addPointCloud<pcl::PointXYZRGB>(candidatePointsPcl.makeShared(), "candidatePoints");

            if(idxMin != -1){
                Eigen::Vector3f targetPosition = {candidatePoints(0,idxMin), candidatePoints(1,idxMin), candidatePoints(2,idxMin)};
                std::vector<Eigen::Matrix4f> tfs;
                arm.checkIk(targetPosition, tfs);
                armVis.draw(tfs[0],
                            tfs[0]*tfs[1],
                            tfs[0]*tfs[1]*tfs[2]);

                viewer.addCoordinateSystem(0.1,Eigen::Affine3f(tfs[0]*tfs[1]*tfs[2]), "endEffectorAxis");
                if(displayGripper){
                    Eigen::Matrix4f toolTf = Eigen::Matrix4f::Identity();
                    toolTf(0,3) = 0.2;
                    Eigen::Matrix4f finalTf = tfs[0]*tfs[1]*tfs[2]*toolTf;

                    pcl::PointCloud<pcl::PointXYZ> newPoints;
                    pcl::fromPCLPointCloud2(gripperModel.cloud, newPoints);
                    for(auto &p: newPoints) { p.x /=1000; p.y /=1000;  p.z /=1000;}
                    pcl::transformPointCloud(newPoints, newPoints, finalTf);
                    viewer.addPolygonMesh<pcl::PointXYZ>(newPoints.makeShared(), gripperModel.polygons, "tool", 0);
                }

            }else{
                std::cout << "Failed to find grasp" << std::endl;
                std::vector<Eigen::Matrix4f> tfs;
                arm.directKinematic({0.0f,0.0f,0.0f}, tfs);
                armVis.draw(tfs[0],
                            tfs[0]*tfs[1],
                            tfs[0]*tfs[1]*tfs[2]);
                viewer.addCoordinateSystem(0.1,Eigen::Affine3f(tfs[0]*tfs[1]*tfs[2]), "endEffectorAxis");
            }
        }else{
            std::cout << "Failed to find grasp" << std::endl;
             std::vector<Eigen::Matrix4f> tfs;
            arm.directKinematic({0.0f,0.0f,0.0f}, tfs);

            armVis.draw(tfs[0],
                        tfs[0]*tfs[1],
                        tfs[0]*tfs[1]*tfs[2]);
            viewer.addCoordinateSystem(0.1,Eigen::Affine3f(tfs[0]*tfs[1]*tfs[2]), "endEffectorAxis");
            
        }

        while(!next && run){
            viewer.spinOnce(30);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        next = false;
    }
}