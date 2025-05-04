import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 1200000, // 20 min
  headers: {
    'Content-Type': 'application/json',
  },
});

// Helper function to handle errors
const handleError = (error) => {
  if (error.response) {
    // The request was made and the server responded with a status code
    console.error('API Error:', error.response.status, error.response.data);
    throw new Error(error.response.data.error || error.response.data.detail || 'API request failed');
  } else if (error.request) {
    // The request was made but no response was received
    console.error('API Error: No response received', error.request);
    throw new Error('No response received from server');
  } else {
    // Something happened in setting up the request
    console.error('API Error:', error.message);
    throw new Error('Error setting up API request');
  }
};

export const analyzeSingleTicket = async (ticketData, quick = false) => {
  try {
    const response = await api.post('/analyze_ticket', ticketData, {
      params: { quick },
    });
    return response.data;
  } catch (error) {
    handleError(error);
  }
};

export const analyzeBulkTickets = async (tickets, quick = false) => {
  try {
    console.log('Input tickets:', tickets);
    // Convert tickets to plain objects and ensure required fields
    const payload = tickets.map(ticket => ({
      subject: ticket.subject || '',
      description: ticket.description || '',
      priority: ticket.priority || 'medium',
      category: ticket.category || null, // Use null instead of undefined
      status: 'new', // Add default status if your backend expects it
      created_at: new Date().toISOString() // Add timestamp if needed
    }));

    console.log('API Payload:', payload);
    console.log(`Calling /bulk_analyze?quick=${quick}`);

    // FIXED: Send payload array directly, not wrapped in a tickets object
    const response = await api.post('/bulk_analyze', 
      payload, // Direct array, not nested in an object
      {
        params: { quick },
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Bulk analysis error:', error.response?.data || error.message);
    handleError(error); // Use the existing error handler
  }
};

export const uploadCSV = async (file, subjectCol, descriptionCol, categoryCol, priorityCol) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('subject_col', subjectCol);
    formData.append('description_col', descriptionCol);
    if (categoryCol) formData.append('category_col', categoryCol);
    if (priorityCol) formData.append('priority_col', priorityCol);

    const response = await api.post('/upload_csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    handleError(error);
  }
};

export const trainClassifier = async (tickets, categories) => {
  try {
    const response = await api.post('/train_classifier', {
      tickets,
      categories,
    });
    return response.data;
  } catch (error) {
    handleError(error);
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('/model_info');
    return response.data;
  } catch (error) {
    handleError(error);
  }
};

export const getProcessingStatus = async () => {
  try {
    const response = await api.get('/processing_status');
    return response.data;
  } catch (error) {
    handleError(error);
  }
};