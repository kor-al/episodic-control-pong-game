#include <iostream>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>


#include <opencv2/opencv.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>


//https://cheind.wordpress.com/2011/12/06/serialization-of-cvmat-objects-using-boost/ // BINARY
//this code is from http://stackoverflow.com/a/21444792/1072039
/** Serialization support for cv::Mat */
namespace boost {
	namespace serialization {
		template<class Archive>
		void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
		{
			int cols, rows, type;
			bool continuous;

			if (Archive::is_saving::value) {
				cols = mat.cols; rows = mat.rows; type = mat.type();
				continuous = mat.isContinuous();
			}

			ar & cols & rows & type & continuous;

			if (Archive::is_loading::value)
				mat.create(rows, cols, type);

			if (continuous) {
				const unsigned int data_size = rows * cols * mat.elemSize();
				ar & boost::serialization::make_array(mat.ptr(), data_size);
			}
			else {
				const unsigned int row_size = cols*mat.elemSize();
				for (int i = 0; i < rows; i++) {
					ar & boost::serialization::make_array(mat.ptr(i), row_size);
				}
			}
		}

	} // namespace serialization
} // namespace boost



////http://stackoverflow.com/questions/16125574/how-to-serialize-opencv-mat-with-boost-xml-archive
//namespace boost {
//	namespace serialization {
//
//
//		template<class Archive>
//		inline void serialize(Archive & ar, cv::Mat& m, const unsigned int version) {
//			int cols = m.cols;
//			int rows = m.rows;
//			size_t elemSize = m.elemSize();
//			size_t elemType = m.type();
//
//			ar & BOOST_SERIALIZATION_NVP(cols);
//			ar & BOOST_SERIALIZATION_NVP(rows);
//			ar & BOOST_SERIALIZATION_NVP(elemSize);
//			ar & BOOST_SERIALIZATION_NVP(elemType); // element type.
//
//			if (m.type() != elemType || m.rows != rows || m.cols != cols) {
//				m = cv::Mat(rows, cols, elemType, cv::Scalar(0));
//			}
//
//			size_t dataSize = cols * rows * elemSize;
//
//			for (size_t dc = 0; dc < dataSize; dc++) {
//				std::stringstream ss;
//				ss << "elem_" << dc;
//				ar & boost::serialization::make_nvp(ss.str().c_str(), m.data[dc]);
//			}
//
//		}
//	}
//}
