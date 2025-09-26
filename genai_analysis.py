import os
import numpy as np
import pydicom
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st
import tempfile
import uuid
import io
import zipfile
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-api-key-here":
    st.error("⚠️ Please set your Google API Key in the GOOGLE_API_KEY variable")
    st.stop()

# Initialize the Medical Agent
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Medical Analysis Queries
series_query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. 
Analyze this medical image series (CT/MRI) and provide comprehensive analysis.
"""

single_image_query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. 
Analyze this medical image and provide comprehensive analysis.
"""

def process_dicom_from_bytes(file_bytes, original_name):
    """Process DICOM file from bytes without using the original filename"""
    try:
        dicom_bytes = io.BytesIO(file_bytes)
        dicom = pydicom.dcmread(dicom_bytes)
        pixel_array = dicom.pixel_array.astype(float)
        
        # Normalize pixel array
        if pixel_array.max() > pixel_array.min():
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
        else:
            pixel_array = pixel_array * 255.0 / pixel_array.max() if pixel_array.max() > 0 else pixel_array
            
        pil_image = PILImage.fromarray(np.uint8(pixel_array)).convert("L")
        
        return {
            'dicom': dicom,
            'pil_image': pil_image,
            'instance_number': getattr(dicom, 'InstanceNumber', 0),
            'slice_location': getattr(dicom, 'SliceLocation', 0),
            'modality': getattr(dicom, 'Modality', 'Unknown'),
            'series_description': getattr(dicom, 'SeriesDescription', 'Unknown'),
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_medical_image(pil_image, is_series=False):
    """Analyze medical image using the AI agent"""
    try:
        # Resize image for analysis
        width, height = pil_image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = pil_image.resize((new_width, new_height))
        
        # Save to temporary file with safe name
        temp_path = f"medical_image_{uuid.uuid4().hex}.png"
        resized_image.save(temp_path)
        
        # Create Agno image
        agno_image = AgnoImage(filepath=temp_path)
        
        # Choose appropriate query
        query = series_query if is_series else single_image_query
        
        # Get analysis
        response = medical_agent.run(query, images=[agno_image])
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return response.content
        
    except Exception as e:
        return f"⚠️ Analysis error: {e}"

def extract_dicom_files_from_zip(zip_file):
    """Extract and process DICOM files from ZIP archive with better detection"""
    dicom_files = []
    all_files = []
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # First, list all files in the ZIP
            file_list = zip_ref.namelist()
            st.sidebar.info(f"📁 ZIP contains {len(file_list)} files")
            
            # Process each file in the ZIP
            for file_info in file_list:
                # Skip directories
                if file_info.endswith('/'):
                    continue
                    
                # Try to read the file
                try:
                    with zip_ref.open(file_info) as file:
                        file_bytes = file.read()
                        
                        # Check if it's a DICOM file by trying to parse it
                        try:
                            # Try to read as DICOM
                            dicom_bytes = io.BytesIO(file_bytes)
                            dicom = pydicom.dcmread(dicom_bytes)
                            
                            # If successful, it's a DICOM file
                            display_name = f"slice_{len(dicom_files) + 1:04d}.dcm"
                            result = process_dicom_from_bytes(file_bytes, display_name)
                            
                            if result['success']:
                                dicom_files.append({
                                    'name': display_name,
                                    'original_name': file_info,
                                    'data': result,
                                    'type': 'dicom'
                                })
                                st.sidebar.success(f"✅ Found DICOM: {file_info}")
                            
                        except (pydicom.errors.InvalidDicomError, AttributeError, EOFError):
                            # Not a DICOM file, skip it
                            all_files.append(file_info)
                            continue
                            
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not read {file_info}: {e}")
                    continue
                    
    except Exception as e:
        st.sidebar.error(f"❌ Error processing ZIP file: {e}")
        return []
    
    # Show summary
    if dicom_files:
        st.sidebar.success(f"🎉 Successfully loaded {len(dicom_files)} DICOM files")
    else:
        st.sidebar.warning(f"🔍 Found {len(all_files)} non-DICOM files in ZIP")
        st.sidebar.info("Files in ZIP: " + ", ".join(all_files[:10]) + ("..." if len(all_files) > 10 else ""))
    
    return dicom_files

# Streamlit UI
st.set_page_config(page_title="Medical Image Analysis", layout="wide")
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown("Upload medical images for AI-powered analysis")

# Initialize session state
if 'medical_images' not in st.session_state:
    st.session_state.medical_images = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = {}

# Solution: Use ZIP file upload to bypass filename restrictions
st.sidebar.header("📁 Upload Medical Images")

st.sidebar.info("""
**For DICOM files with numeric names:**
1. **ZIP File (Recommended)**: Compress your DICOM files into a ZIP archive
2. **Rename Files**: Add a prefix like 'slice_' to numeric filenames
""")

# Option 1: ZIP file upload (bypasses filename restrictions)
st.sidebar.subheader("🎯 Option 1: Upload ZIP File (Recommended)")
zip_file = st.sidebar.file_uploader(
    "Upload a ZIP file containing DICOM images",
    type=['zip'],
    key="zip_upload",
    accept_multiple_files=False,
    help="Create a ZIP file of your DICOM series to avoid filename issues"
)

# Option 2: Single file upload with renamed files
st.sidebar.subheader("📄 Option 2: Upload Individual Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload individual medical image files",
    type=['dcm', 'jpg', 'jpeg', 'png', 'bmp'],
    key="file_upload",
    accept_multiple_files=True,
    help="If you get errors, rename numeric filenames (e.g., '000001.dcm' → 'slice_000001.dcm')"
)

# Process ZIP file upload
if zip_file is not None:
    with st.spinner("🔍 Scanning ZIP file for DICOM images..."):
        dicom_files = extract_dicom_files_from_zip(zip_file)
        if dicom_files:
            st.session_state.medical_images = dicom_files
        else:
            st.error("""
            ## ❌ No DICOM files found in the ZIP archive
            
            **Possible reasons:**
            1. The ZIP file doesn't contain .dcm files
            2. The DICOM files are in subfolders
            3. The files are not valid DICOM format
            
            **Solutions:**
            - Ensure your ZIP contains actual DICOM (.dcm) files
            - Try uploading individual files using Option 2
            - Make sure files are not corrupted
            """)

# Process individual file uploads
elif uploaded_files and len(uploaded_files) > 0:
    processed_files = []
    
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.getvalue()
            original_name = uploaded_file.name
            
            # Create a safe display name
            if original_name.replace('.dcm', '').isdigit():
                display_name = f"slice_{original_name}"
            else:
                display_name = original_name
            
            if uploaded_file.name.lower().endswith('.dcm'):
                result = process_dicom_from_bytes(file_bytes, display_name)
                if result['success']:
                    processed_files.append({
                        'name': display_name,
                        'original_name': original_name,
                        'data': result,
                        'type': 'dicom'
                    })
                    st.sidebar.success(f"✅ Loaded: {original_name}")
                else:
                    st.sidebar.error(f"❌ Failed to process: {original_name}")
            else:
                # Process standard image
                pil_image = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
                processed_files.append({
                    'name': display_name,
                    'original_name': original_name,
                    'pil_image': pil_image,
                    'type': 'standard'
                })
                st.sidebar.success(f"✅ Loaded: {original_name}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error processing {uploaded_file.name}: {e}")
    
    if processed_files:
        st.session_state.medical_images = processed_files

# Display and analyze images
if st.session_state.medical_images:
    st.header("📊 Image Analysis")
    
    # Separate DICOM from standard images
    dicom_images = [img for img in st.session_state.medical_images if img['type'] == 'dicom']
    standard_images = [img for img in st.session_state.medical_images if img['type'] == 'standard']
    
    # Handle DICOM images
    if dicom_images:
        st.subheader("🩻 DICOM Images")
        
        # Sort DICOM images by instance number
        dicom_images.sort(key=lambda x: x['data']['instance_number'])
        
        if len(dicom_images) == 1:
            # Single DICOM image
            img_data = dicom_images[0]
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_data['data']['pil_image'], 
                       caption=f"DICOM: {img_data['name']}", 
                       use_container_width=True)
            
            with col2:
                st.write("**DICOM Information:**")
                st.write(f"**Modality:** {img_data['data']['modality']}")
                st.write(f"**Series:** {img_data['data']['series_description']}")
                st.write(f"**Instance:** {img_data['data']['instance_number']}")
                st.write(f"**Slice Location:** {img_data['data']['slice_location']}")
                st.write(f"**Original File:** {img_data['original_name']}")
                
                if st.button("Analyze This Image", key=f"analyze_single_{img_data['name']}"):
                    with st.spinner("🔄 Analyzing image..."):
                        analysis = analyze_medical_image(img_data['data']['pil_image'])
                        st.session_state.analysis_done[img_data['name']] = analysis
                        st.rerun()
        
        else:
            # Multiple DICOM images - series view
            st.success(f"📚 DICOM Series Detected: {len(dicom_images)} slices")
            
            # Show series info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Slices:** {len(dicom_images)}")
                st.write(f"**Modality:** {dicom_images[0]['data']['modality']}")
            with col2:
                st.write(f"**Series Description:** {dicom_images[0]['data']['series_description']}")
                st.write(f"**Files from:** {dicom_images[0]['original_name'].split('/')[0] if '/' in dicom_images[0]['original_name'] else 'ZIP archive'}")
            
            # Series navigation
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("**Series Navigation**")
                slice_idx = st.slider("Select Slice", 1, len(dicom_images), 1, key="slice_nav") - 1
                
                # Quick navigation
                st.write("**Quick Access:**")
                cols = st.columns(3)
                positions = [0, len(dicom_images)//2, len(dicom_images)-1]
                for i, pos in enumerate(positions):
                    with cols[i]:
                        if st.button(f"Slice {pos+1}", key=f"nav_{i}"):
                            slice_idx = pos
                            st.rerun()
            
            with col2:
                current_slice = dicom_images[slice_idx]
                st.image(current_slice['data']['pil_image'],
                       caption=f"Slice {slice_idx+1}/{len(dicom_images)} - {current_slice['data']['modality']}",
                       use_container_width=True)
                
                st.write(f"**Instance:** {current_slice['data']['instance_number']} | "
                        f"**Location:** {current_slice['data']['slice_location']}")
                st.write(f"**File:** {current_slice['original_name']}")
            
            # Analysis options for series
            st.subheader("🔍 Analysis Options")
            analysis_mode = st.radio(
                "Choose analysis scope:",
                ["Current slice only", "Key slices (sample)", "Complete series overview"],
                horizontal=True
            )
            
            if st.button("🚀 Start Analysis", type="primary", key="analyze_series"):
                with st.spinner("🔄 Analyzing medical images..."):
                    if analysis_mode == "Current slice only":
                        analysis = analyze_medical_image(current_slice['data']['pil_image'], is_series=True)
                        st.session_state.analysis_done['current_slice'] = analysis
                    
                    elif analysis_mode == "Key slices (sample)":
                        # Analyze representative slices
                        sample_indices = [0, len(dicom_images)//4, len(dicom_images)//2, 
                                        (3*len(dicom_images))//4, len(dicom_images)-1]
                        combined_analysis = "# 📋 Series Analysis Report\n\n"
                        
                        progress_bar = st.progress(0)
                        for i, idx in enumerate(sample_indices):
                            if idx < len(dicom_images):
                                sample_slice = dicom_images[idx]
                                analysis = analyze_medical_image(sample_slice['data']['pil_image'], is_series=True)
                                combined_analysis += f"## Slice {idx+1}/{len(dicom_images)}\n\n{analysis}\n\n---\n\n"
                                progress_bar.progress((i + 1) / len(sample_indices))
                        
                        st.session_state.analysis_done['series_analysis'] = combined_analysis
                    
                    else:
                        # Quick overview of first few slices
                        max_slices = min(3, len(dicom_images))
                        overview_analysis = "# 📋 Series Overview\n\n"
                        
                        for i in range(max_slices):
                            slice_data = dicom_images[i]
                            analysis = analyze_medical_image(slice_data['data']['pil_image'], is_series=True)
                            overview_analysis += f"## Slice {i+1}\n\n{analysis}\n\n"
                        
                        st.session_state.analysis_done['series_overview'] = overview_analysis
                    
                    st.rerun()
    
    # Handle standard images
    if standard_images:
        st.subheader("🖼️ Standard Images")
        for img_data in standard_images:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_data['pil_image'],
                       caption=f"Image: {img_data['name']}",
                       use_container_width=True)
            
            with col2:
                if st.button(f"Analyze {img_data['name']}", key=f"analyze_std_{img_data['name']}"):
                    with st.spinner("🔄 Analyzing image..."):
                        analysis = analyze_medical_image(img_data['pil_image'])
                        st.session_state.analysis_done[img_data['name']] = analysis
                        st.rerun()

    # Display analysis results
    if st.session_state.analysis_done:
        st.header("📋 Analysis Results")
        
        for key, analysis in st.session_state.analysis_done.items():
            with st.expander(f"📄 Analysis Report: {key}", expanded=True):
                st.markdown(analysis, unsafe_allow_html=True)
        
        if st.button("Clear Results"):
            st.session_state.analysis_done = {}
            st.rerun()

else:
    # Instructions when no files are uploaded
    st.info("""
    ## 📋 How to Use This Tool
    
    ### For DICOM Files (CT/MRI):
    **Option 1 (Recommended):** 
    - Compress your DICOM files into a ZIP archive
    - Upload the ZIP file using Option 1 above
    
    **Option 2:**
    - Rename files with numeric names (e.g., `000001.dcm` → `slice_000001.dcm`)
    - Upload individual files using Option 2
    
    ### For Standard Images (X-ray, Ultrasound):
    - Upload directly using Option 2
    
    ### Supported Formats:
    - **DICOM**: .dcm files (CT, MRI scans)
    - **Images**: .jpg, .jpeg, .png, .bmp (X-rays, ultrasounds)
    """)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool is for educational and research purposes only. 
Always consult qualified healthcare professionals for medical diagnoses.
""")

# Clear all data button
if st.sidebar.button("🔄 Clear All Data"):
    st.session_state.medical_images = []
    st.session_state.analysis_done = {}
    st.rerun()