-- phpMyAdmin SQL Dump
-- version 4.8.4
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Mar 31, 2019 at 09:15 AM
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
  `lec_time` timestamp NOT NULL,
  PRIMARY KEY (`lec_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `attendance_marking_faculty`
--

DROP TABLE IF EXISTS `attendance_marking_faculty`;
CREATE TABLE IF NOT EXISTS `attendance_marking_faculty` (
  `f_id` int(2) NOT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `date1` date NOT NULL,
  `lec_time` timestamp NOT NULL,
  PRIMARY KEY (`f_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `faculty`
--

DROP TABLE IF EXISTS `faculty`;
CREATE TABLE IF NOT EXISTS `faculty` (
  `f_id` int(2) NOT NULL,
  `class` varchar(7) DEFAULT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `fname` varchar(20) DEFAULT NULL,
  `lname` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`f_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

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
  `fname` varchar(20) DEFAULT NULL,
  `lname` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`lec_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `student`
--

DROP TABLE IF EXISTS `student`;
CREATE TABLE IF NOT EXISTS `student` (
  `roll_no` int(2) NOT NULL,
  `class` varchar(7) DEFAULT NULL,
  `branch` varchar(4) DEFAULT NULL,
  `fname` varchar(20) DEFAULT NULL,
  `lname` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`roll_no`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `student`
--

INSERT INTO `student` (`roll_no`, `class`, `branch`, `fname`, `lname`) VALUES
(3, 'TE', 'Comp', 'Aditya', 'Aspat'),
(34, 'TE', 'Comp', 'Sohum', 'Khot'),
(37, 'TE', 'Comp', 'Shivang', 'Medhekar');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
