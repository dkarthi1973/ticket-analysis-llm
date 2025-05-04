import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  CircularProgress,
  Paper,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
} from '@mui/material';
import { uploadCSV } from '../services/api';
import { useSnackbar } from 'notistack';
import { FileUpload } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const TicketUpload = () => {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [subjectCol, setSubjectCol] = useState('subject');
  const [descriptionCol, setDescriptionCol] = useState('description');
  const [categoryCol, setCategoryCol] = useState('');
  const [priorityCol, setPriorityCol] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const { enqueueSnackbar } = useSnackbar();
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setUploadResult(null);
      setError(null);
      
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const lines = text.split('\n');
        if (lines.length > 0) {
          const headers = lines[0].split(',');
          setColumns(headers);
          if (headers.includes('subject')) setSubjectCol('subject');
          if (headers.includes('description')) setDescriptionCol('description');
          if (headers.includes('category')) setCategoryCol('category');
          if (headers.includes('priority')) setPriorityCol('priority');
        }
      };
      reader.readAsText(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      enqueueSnackbar('Please select a file first', { variant: 'warning' });
      return;
    }

    if (!subjectCol || !descriptionCol) {
      enqueueSnackbar('Subject and Description columns are required', { variant: 'error' });
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const result = await uploadCSV(
        file,
        subjectCol,
        descriptionCol,
        categoryCol || undefined,
        priorityCol || undefined
      );
      
      setUploadResult(result);
      
      // Store the uploaded tickets in localStorage for BulkAnalysis component
      if (result && result.tickets && result.tickets.length > 0) {
        localStorage.setItem('csvTickets', JSON.stringify(result.tickets));
      }
      
      enqueueSnackbar('File uploaded successfully!', { variant: 'success' });
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.message || 'Failed to upload file');
      enqueueSnackbar(error.message || 'Failed to upload file', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleGoToBulkAnalysis = () => {
    if (uploadResult && uploadResult.tickets) {
      // Make sure the tickets are in localStorage before navigating
      localStorage.setItem('csvTickets', JSON.stringify(uploadResult.tickets));
      
      // Navigate to the correct route - "/bulk" instead of "/bulk-analysis"
      navigate('/bulk');
    } else {
      enqueueSnackbar('No tickets available to analyze', { variant: 'warning' });
    }
  };

  const sentimentColor = (score) => {
    if (score >= 0.5) return 'success';
    if (score >= 0.1) return 'info';
    if (score <= -0.5) return 'error';
    if (score <= -0.1) return 'warning';
    return 'default';
  };

  const priorityColor = (priority) => {
    if (!priority) return 'default';
    switch (priority.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        CSV Ticket Upload
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload CSV File
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <Button
                  variant="contained"
                  component="label"
                  startIcon={<FileUpload />}
                >
                  Select CSV File
                  <input
                    type="file"
                    accept=".csv"
                    hidden
                    onChange={handleFileChange}
                  />
                </Button>
                {file && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Selected: {file.name}
                  </Typography>
                )}
              </Box>
              
              {columns.length > 0 && (
                <>
                  <Typography variant="subtitle2" gutterBottom>
                    Map CSV Columns
                  </Typography>
                  
                  <TextField
                    select
                    fullWidth
                    label="Subject Column"
                    value={subjectCol}
                    onChange={(e) => setSubjectCol(e.target.value)}
                    margin="normal"
                    SelectProps={{
                      native: true,
                    }}
                  >
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </TextField>
                  
                  <TextField
                    select
                    fullWidth
                    label="Description Column"
                    value={descriptionCol}
                    onChange={(e) => setDescriptionCol(e.target.value)}
                    margin="normal"
                    SelectProps={{
                      native: true,
                    }}
                  >
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </TextField>
                  
                  <TextField
                    select
                    fullWidth
                    label="Category Column (optional)"
                    value={categoryCol}
                    onChange={(e) => setCategoryCol(e.target.value)}
                    margin="normal"
                    SelectProps={{
                      native: true,
                    }}
                  >
                    <option value="">None</option>
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </TextField>
                  
                  <TextField
                    select
                    fullWidth
                    label="Priority Column (optional)"
                    value={priorityCol}
                    onChange={(e) => setPriorityCol(e.target.value)}
                    margin="normal"
                    SelectProps={{
                      native: true,
                    }}
                  >
                    <option value="">None</option>
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </TextField>
                  
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={handleUpload}
                      disabled={loading || !file}
                      startIcon={loading ? <CircularProgress size={20} /> : null}
                    >
                      {loading ? 'Uploading...' : 'Upload and Analyze'}
                    </Button>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          {uploadResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Upload Results
                </Typography>
                
                <Typography variant="body1" paragraph>
                  Successfully loaded {uploadResult.tickets.length} tickets from CSV.
                </Typography>
                
                {uploadResult.sample_analysis && uploadResult.sample_analysis.length > 0 && (
                  <>
                    <Typography variant="subtitle2" gutterBottom>
                      Sample Analysis (First {uploadResult.sample_analysis.length} tickets)
                    </Typography>
                    
                    <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
                      <Table size="small" stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell>Subject</TableCell>
                            <TableCell>Category</TableCell>
                            <TableCell>Priority</TableCell>
                            <TableCell>Sentiment</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {uploadResult.sample_analysis.map((result, index) => (
                            <TableRow key={index}>
                              <TableCell>{uploadResult.tickets[index]?.subject}</TableCell>
                              <TableCell>{result.category}</TableCell>
                              <TableCell>
                                <Chip
                                  label={result.priority}
                                  color={priorityColor(result.priority)}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={result.sentiment.toFixed(2)}
                                  color={sentimentColor(result.sentiment)}
                                  size="small"
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="outlined"
                        color="primary"
                        onClick={handleGoToBulkAnalysis}
                      >
                        Proceed to Bulk Analysis
                      </Button>
                    </Box>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default TicketUpload;