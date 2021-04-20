<?php
  session_start();
  include ("Database/dbconfig.php");
  {
    $stud = $_SESSION['student'];
    $sq = "SELECT * from student WHERE stud_id = '$stud'";
    $esq=mysqli_query($conn,$sq);
    $row=mysqli_fetch_assoc($esq);
    $fname = $row["fname"];
    $sname = $row["lname"];
    $roll = $row["roll_no"];
    $querystudspcc = $conn->query("SELECT * FROM attendance_marking WHERE roll_no = '$roll' AND lec_id = 1");
    $sspcc = $querystudspcc->num_rows;
    $querystudcss = $conn->query("SELECT * FROM attendance_marking WHERE roll_no = '$roll' AND lec_id = 2");
    $scss = $querystudcss->num_rows;
    $querystudml = $conn->query("SELECT * FROM attendance_marking WHERE roll_no = '$roll' AND lec_id = 3");
    $sml = $querystudml->num_rows;
    $querystuddwm = $conn->query("SELECT * FROM attendance_marking WHERE roll_no = '$roll' AND lec_id = 4");
    $sdwm = $querystuddwm->num_rows;
    $querystudse = $conn->query("SELECT * FROM attendance_marking WHERE roll_no = '$roll' AND lec_id = 5");
    $sse = $querystudse->num_rows;
    $queryspcc = $conn->query("SELECT * FROM attendance_marking_faculty WHERE lec_id = 1");
    $spcc = $queryspcc->num_rows;
    $querycss = $conn->query("SELECT * FROM attendance_marking_faculty WHERE lec_id = 2");
    $css = $querycss->num_rows;
    $queryml = $conn->query("SELECT * FROM attendance_marking_faculty WHERE lec_id = 3");
    $ml= $queryml->num_rows;
    $querydwm = $conn->query("SELECT * FROM attendance_marking_faculty WHERE lec_id = 4");
    $dwm = $querydwm->num_rows;
    $queryse = $conn->query("SELECT * FROM attendance_marking_faculty WHERE lec_id = 5");
    $se = $queryse->num_rows;
    $aspcc = ($sspcc / $spcc) * 100;
    $acss = ($scss / $css) * 100;
    $aml = ($sml / $ml) * 100;
    $adwm = ($sdwm / $dwm) * 100;
    $ase = ($sse / $se) * 100;
    #$equery2 = mysqli_query($conn,$query2);
    #row2 = mysqli_fetch_assoc($equery2);
    #$lec_id = $row2["lec_id"];
    #$branch = $row2["branch"];
  }
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <meta name="description" content="">
  <meta name="author" content="">
  <title>Student Attendance</title>
  <!-- Custom fonts for this template-->
  <link href="vendor/fontawesome-free/css/all.min.css" rel="stylesheet" type="text/css">
  <link href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i" rel="stylesheet">
  <!-- Custom styles for this template-->
  <link href="css/sb-admin-2.min.css" rel="stylesheet">
</head>
<body id="page-top">
  <!-- Page Wrapper -->
  <div id="wrapper">
    <!-- Sidebar -->
    <ul class="navbar-nav bg-gradient-primary sidebar sidebar-dark accordion" id="accordionSidebar">
      <!-- Sidebar - Brand -->
      <a class="sidebar-brand d-flex align-items-center justify-content-center" href="index.php">
        <div class="sidebar-brand-icon">
          <i class="fas fa-university"></i>
        </div>
        <div class="sidebar-brand-text mx-3">XIE - AMS</div>
      </a>

      <!-- Divider -->
      <hr class="sidebar-divider my-0">

      <!-- Nav Item - Dashboard -->
      <li class="nav-item active">
        <a class="nav-link" href="index.php">
          <span>Student Information</span></a>
      </li>

      <!-- Divider -->
      <hr class="sidebar-divider">
      <!-- Sidebar Toggler (Sidebar) -->
      <div class="text-center d-none d-md-inline">
        <button class="rounded-circle border-0" id="sidebarToggle"></button>
      </div>
    </ul>
    <!-- End of Sidebar -->
    <!-- Content Wrapper -->
    <div id="content-wrapper" class="d-flex flex-column">
      <!-- Main Content -->
      <div id="content">
        <!-- Topbar -->
        <nav class="navbar navbar-expand navbar-light bg-white topbar mb-4 static-top shadow">
          <!-- Sidebar Toggle (Topbar) -->
          <button id="sidebarToggleTop" class="btn btn-link d-md-none rounded-circle mr-3">
            <i class="fa fa-bars"></i>
          </button>
          <ul class="navbar-nav ml-auto">
            <div class="topbar-divider d-none d-sm-block"></div>
            <!-- Nav Item - User Information -->
            <li class="nav-item dropdown no-arrow">
              <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                <span class="mr-2 d-none d-lg-inline text-gray-800 big"><?php echo $fname; ?></span>
                <img class="img-profile rounded-circle" src="img/college.png">
              </a>
              <!-- Dropdown - User Information -->
              <div class="dropdown-menu dropdown-menu-right shadow animated--grow-in" aria-labelledby="userDropdown">
                <a class="dropdown-item" href="#" data-toggle="modal" data-target="#logoutModal">
                  <i class="fas fa-sign-out-alt fa-sm fa-fw mr-2 text-gray-400"></i>
                  Logout
                </a>
              </div>
            </li>
          </ul>
        </nav>
        <!-- End of Topbar -->
        <!-- Begin Page Content -->
        <div class="container-fluid">
          <!-- Page Heading -->
          <div class="d-sm-flex align-items-center justify-content-between mb-4">
            <h1 class="h2 mb-0 text-gray-800" align="center"><?php echo $fname; ?>, your attendance : </h1>
          </div>
          <div class="row">
            <div class="col-lg-10 offset-1 mb-10">
              <!-- Project Card Example -->
              <div class="card shadow mb-4">
                <div class="card-header py-3">
                  <h6 class="m-0 font-weight-bold text-primary">Lectures</h6>
                </div>
                <div class="card-body">
                  <h4 class="small font-weight-bold">SPCC<span class="float-right"><?php echo number_format((float)$aspcc,2,'.',''); ?>%</span></h4>
                  <div class="progress mb-4">
                    <?php echo "<div class='progress-bar bg-danger' role='progressbar' style='width:".$aspcc."%' aria-valuenow='20' aria-valuemin='0' aria-valuemax='100'></div>";?>
                    <!-- <div class="progress-bar bg-danger" role="progressbar" style="width:60%" aria-valuenow="20" aria-valuemin="0" aria-valuemax="100"></div> -->
                  </div>
                  <h4 class="small font-weight-bold">CSS<span class="float-right"><?php echo number_format((float)$acss,2,'.',''); ?>%</span></h4>
                  <div class="progress mb-4">
                    <?php echo "<div class='progress-bar bg-warning' role='progressbar' style='width:".$acss."%' aria-valuenow='20' aria-valuemin='0' aria-valuemax='100'></div>";?>
                    <!-- <div class="progress-bar bg-warning" role="progressbar" style="width: 40%" aria-valuenow="40" aria-valuemin="0" aria-valuemax="100"></div> -->
                  </div>
                  <h4 class="small font-weight-bold">ML<span class="float-right"><?php echo number_format((float)$aml,2,'.',''); ?>%</span></h4>
                  <div class="progress mb-4">
                    <?php echo "<div class='progress-bar' role='progressbar' style='width:".$aml."%' aria-valuenow='20' aria-valuemin='0' aria-valuemax='100'></div>";?>
                    <!-- <div class="progress-bar" role="progressbar" style="width: 60%" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100"></div> -->
                  </div>
                  <h4 class="small font-weight-bold">DWM<span class="float-right"><?php echo number_format((float)$adwm,2,'.',''); ?>%</span></h4>
                  <div class="progress mb-4">
                    <?php echo "<div class='progress-bar bg-info' role='progressbar' style='width:".$adwm."%' aria-valuenow='20' aria-valuemin='0' aria-valuemax='100'></div>";?>
                    <!-- <div class="progress-bar bg-info" role="progressbar" style="width: 80%" aria-valuenow="80" aria-valuemin="0" aria-valuemax="100"></div> -->
                  </div>
                  <h4 class="small font-weight-bold">SE<span class="float-right"><?php echo number_format((float)$ase,2,'.',''); ?>%</span></h4>
                  <div class="progress">
                    <?php echo "<div class='progress-bar bg-success' role='progressbar' style='width:".$ase."%' aria-valuenow='20' aria-valuemin='0' aria-valuemax='100'></div>";?>
                    <!-- <div class="progress-bar bg-success" role="progressbar" style="width: 100%" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100"></div> -->
                  </div>
                </div>
              </div>
            </div>
          </div>

        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

      <!-- Footer -->
      <br><br><br><br><br><br><br><br><br><br><br><br><br><br>
      <footer class="sticky-footer bg-white">
        <div class="container my-auto">
          <div class="copyright text-center my-auto">
            <span>Copyright &copy; SAS - 2019</span>
          </div>
        </div>
      </footer>
      <!-- End of Footer -->

    </div>
    <!-- End of Content Wrapper -->

  </div>
  <!-- End of Page Wrapper -->

  <!-- Scroll to Top Button-->
  <a class="scroll-to-top rounded" href="#page-top">
    <i class="fas fa-angle-up"></i>
  </a>

  <!-- Logout Modal-->
  <div class="modal fade" id="logoutModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="exampleModalLabel">Confirm Logout?</h5>
          <button class="close" type="button" data-dismiss="modal" aria-label="Close">
            <span aria-hidden="true">×</span>
          </button>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" type="button" data-dismiss="modal">Cancel</button>
          <a class="btn btn-primary" href="logout.php">Logout</a>
        </div>
      </div>
    </div>
  </div>

  <!-- Bootstrap core JavaScript-->
  <script src="vendor/jquery/jquery.min.js"></script>
  <script src="vendor/bootstrap/js/bootstrap.bundle.min.js"></script>

  <!-- Core plugin JavaScript-->
  <script src="vendor/jquery-easing/jquery.easing.min.js"></script>

  <!-- Custom scripts for all pages-->
  <script src="js/sb-admin-2.min.js"></script>

  <!-- Page level plugins -->
  <script src="vendor/chart.js/Chart.min.js"></script>

  <!-- Page level custom scripts -->
  <script src="js/demo/chart-area-demo.js"></script>
  <script src="js/demo/chart-pie-demo.js"></script>

</body>

</html>
