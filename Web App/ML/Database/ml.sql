-- phpMyAdmin SQL Dump
-- version 4.8.4
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Apr 02, 2019 at 03:46 AM
-- Server version: 5.7.24
-- PHP Version: 7.2.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ml`
--

-- --------------------------------------------------------

--
-- Table structure for table `attendance_marking`
--

DROP TABLE IF EXISTS `attendance_marking`;
CREATE TABLE IF NOT EXISTS `attendance_marking` (
  `lec_id` int(2) NOT NULL,
  `f_id` int(2) DEFAULT NULL,
  `roll_no` int(2) DEFAULT NULL,
  `class` varchar(7) DEFAULT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `lec_date` date NOT NULL,
  `lec_time` time NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `attendance_marking`
--

INSERT INTO `attendance_marking` (`lec_id`, `f_id`, `roll_no`, `class`, `branch`, `lec_date`, `lec_time`) VALUES
(0, 1, 3, 'TE', 'Comp', '2019-04-01', '09:00:00'),
(1, 1, 3, 'TE', 'Comp', '2019-04-01', '08:00:00'),
(1, 1, 3, 'TE', 'Comp', '2019-03-29', '14:00:00'),
(2, 2, 3, 'TE', 'Comp', '2019-04-01', '09:00:00'),
(2, 2, 3, 'TE', 'Comp', '2019-03-29', '15:00:00'),
(1, 1, 3, 'TE', 'Comp', '2019-04-01', '05:00:00'),
(1, 1, 3, 'TE', 'Comp', '2019-04-01', '05:00:00'),
(1, 1, 3, 'TE', 'Comp', '2019-04-06', '05:00:00'),
(2, 2, 3, 'TE', 'Comp', '2019-04-02', '06:00:00'),
(2, 2, 3, 'TE', 'Comp', '2019-04-02', '06:00:00'),
(5, 5, 3, 'TE', 'Comp', '2019-04-02', '14:00:00'),
(4, 4, 3, 'TE', 'Comp', '2019-04-02', '14:00:00'),
(4, 4, 3, 'TE', 'Comp', '2019-04-02', '14:00:00'),
(4, 4, 3, 'TE', 'Comp', '2019-04-02', '14:00:00'),
(3, 3, 3, 'TE', 'Comp', '2019-04-04', '18:00:00'),
(3, 3, 3, 'TE', 'Comp', '2019-04-04', '18:00:00'),
(3, 3, 3, 'TE', 'Comp', '2019-04-04', '18:00:00'),
(3, 3, 3, 'TE', 'Comp', '2019-04-04', '18:00:00'),
(3, 3, 3, 'TE', 'Comp', '2019-04-04', '18:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `attendance_marking_faculty`
--

DROP TABLE IF EXISTS `attendance_marking_faculty`;
CREATE TABLE IF NOT EXISTS `attendance_marking_faculty` (
  `f_id` int(2) NOT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `enter_date` date NOT NULL,
  `enter_time` time NOT NULL,
  `exit_time` time NOT NULL,
  `lec_id` int(2) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `attendance_marking_faculty`
--

INSERT INTO `attendance_marking_faculty` (`f_id`, `branch`, `enter_date`, `enter_time`, `exit_time`, `lec_id`) VALUES
(1, 'Comp', '2019-04-01', '08:00:00', '00:00:00', 1),
(1, 'Comp', '2019-03-29', '14:00:00', '00:00:00', 1),
(1, 'Comp', '2019-03-28', '15:00:00', '00:00:00', 1),
(1, 'Comp', '2019-03-28', '15:00:00', '00:00:00', 1),
(2, 'Comp', '2019-04-01', '07:00:00', '00:00:00', 2),
(2, 'Comp', '2019-04-01', '08:00:00', '00:00:00', 2),
(2, 'Comp', '2019-04-02', '07:00:00', '00:00:00', 2),
(2, 'Comp', '2019-04-03', '07:00:00', '00:00:00', 2),
(2, 'Comp', '2019-04-04', '07:00:00', '00:00:00', 2),
(2, 'Comp', '2019-04-05', '07:00:00', '00:00:00', 2),
(3, 'Comp', '2019-04-01', '11:00:00', '00:00:00', 3),
(3, 'Comp', '2019-04-01', '11:00:00', '00:00:00', 3),
(3, 'Comp', '2019-04-02', '12:00:00', '00:00:00', 3),
(3, 'Comp', '2019-04-03', '13:00:00', '00:00:00', 3),
(3, 'Comp', '2019-04-04', '14:00:00', '00:00:00', 3),
(3, 'Comp', '2019-04-05', '15:00:00', '00:00:00', 3),
(4, 'Comp', '2019-04-01', '01:00:00', '00:00:00', 4),
(4, 'Comp', '2019-04-01', '01:00:00', '00:00:00', 4),
(4, 'Comp', '2019-04-02', '02:00:00', '00:00:00', 4),
(4, 'Comp', '2019-04-03', '03:00:00', '00:00:00', 4),
(4, 'Comp', '2019-04-04', '04:00:00', '00:00:00', 4),
(4, 'Comp', '2019-04-05', '05:00:00', '00:00:00', 4),
(5, 'Comp', '2019-04-01', '18:00:00', '00:00:00', 5),
(5, 'Comp', '2019-04-01', '18:00:00', '00:00:00', 5),
(5, 'Comp', '2019-04-01', '18:00:00', '00:00:00', 5),
(5, 'Comp', '2019-04-01', '18:00:00', '00:00:00', 5),
(1, 'Comp', '2019-04-06', '00:00:00', '00:00:00', 1),
(1, 'Comp', '2019-04-06', '00:00:00', '00:00:00', 1),
(1, 'Comp', '2019-04-07', '00:00:00', '00:00:00', 1);

-- --------------------------------------------------------

--
-- Table structure for table `faculty`
--

DROP TABLE IF EXISTS `faculty`;
CREATE TABLE IF NOT EXISTS `faculty` (
  `f_id` int(2) NOT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `fname` varchar(20) DEFAULT NULL,
  `lname` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`f_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `faculty`
--

INSERT INTO `faculty` (`f_id`, `branch`, `fname`, `lname`) VALUES
(1, 'CMPN', 'Saurabh', 'Patil'),
(2, 'CMPN', 'Kunal', 'Meher'),
(3, 'CMPN', 'Omprakash', 'Yadav'),
(4, 'CMPN', 'Kavita', 'Jain'),
(5, 'CMPN', 'Nilambari', 'Narkar');

-- --------------------------------------------------------

--
-- Table structure for table `faculty_login`
--

DROP TABLE IF EXISTS `faculty_login`;
CREATE TABLE IF NOT EXISTS `faculty_login` (
  `f_id` int(2) NOT NULL,
  `username` varchar(20) DEFAULT NULL,
  `pass` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`f_id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `lecture`
--

DROP TABLE IF EXISTS `lecture`;
CREATE TABLE IF NOT EXISTS `lecture` (
  `lec_id` int(2) NOT NULL,
  `f_id` int(2) DEFAULT NULL,
  `class` varchar(7) DEFAULT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `lectname` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`lec_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `lecture`
--

INSERT INTO `lecture` (`lec_id`, `f_id`, `class`, `branch`, `lectname`) VALUES
(1, 1, 'TE', 'CMPN', 'SPCC'),
(2, 2, 'TE', 'Comp', 'CSS'),
(3, 3, 'TE', 'Comp', 'ML'),
(4, 4, 'TE', 'Comp', 'DWM'),
(5, 5, 'TE', 'Comp', 'SE');

-- --------------------------------------------------------

--
-- Table structure for table `student`
--

DROP TABLE IF EXISTS `student`;
CREATE TABLE IF NOT EXISTS `student` (
  `roll_no` int(2) NOT NULL,
  `class` varchar(7) DEFAULT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `face_id` int(11) NOT NULL,
  `fname` varchar(20) DEFAULT NULL,
  `lname` varchar(20) DEFAULT NULL,
  `stud_id` varchar(15) NOT NULL,
  `pass` varchar(100) NOT NULL,
  PRIMARY KEY (`roll_no`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `student`
--

INSERT INTO `student` (`roll_no`, `class`, `branch`, `face_id`, `fname`, `lname`, `stud_id`, `pass`) VALUES
(3, 'TE', 'Comp', 0, 'Aditya', 'Aspat', 'XIECE161704', 'adityaaspat'),
(34, 'TE', 'Comp', 0, 'Sohum', 'Khot', 'XIECE161734', 'sohumkhot'),
(37, 'TE', 'Comp', 0, 'Shivang', 'Medhekar', 'XIECE161737', 'shivangmedhekar');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
