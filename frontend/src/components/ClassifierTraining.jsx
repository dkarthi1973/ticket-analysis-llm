import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  CircularProgress,
  Alert,
  Grid,
  Chip,
} from '@mui/material';
import { trainClassifier } from '../services/api';
import { useSnackbar } from 'notistack';
import { FileUpload } from '@mui/icons-material';

const ClassifierTraining = () => {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [subjectCol, setSubjectCol] = useState('subject');
  const [descriptionCol, setDescriptionCol] = useState('description');
  const [categoryCol, setCategoryCol] = useState('category');
  const [loading, setLoading] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const { enqueueSnackbar } = useSnackbar();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setTrainingResult(null);
      
      // Preview the file to get columns
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const lines = text.split('\n');
        if (lines.length > 0) {
          const headers = lines[0].split(',');
          setColumns(headers);
          // Set default values if standard column names exist
          if (headers.includes('subject')) setSubjectCol('subject');
          if (headers.includes('description')) setDescriptionCol('description');
          if (headers.includes('category')) setCategoryCol('category');
        }
      };
      reader.readAsText(selectedFile);
    }
  };

  const handleTrain = async () => {
    if (!file) {
      enqueueSnackbar('Please select a file first', { variant: 'warning' });
      return;
    }

    if (!subjectCol || !descriptionCol || !categoryCol) {
      enqueueSnackbar('Subject, Description, and Category columns are required', { variant: 'error' });
      return;
    }

    setLoading(true);
    try {
      // Read and parse the CSV file
      const reader = new FileReader();
      reader.onload = async (event) => {
        const text = event.target.result;
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        
        const subjectIndex = headers.indexOf(subjectCol);
        const descriptionIndex = headers.indexOf(descriptionCol);
        const categoryIndex = headers.indexOf(categoryCol);
        
        if (subjectIndex === -1 || descriptionIndex === -1 || categoryIndex === -1) {
          throw new Error('Could not find specified columns in the CSV file');
        }
        
        const tickets = [];
        for (let i = 1; i < lines.length; i++) {
          if (!lines[i].trim()) continue;
          
          const values = lines[i].split(',');
          if (values.length >= Math.max(subjectIndex, descriptionIndex, categoryIndex) + 1) {
            tickets.push({
              subject: values[subjectIndex],
              description: values[descriptionIndex],
              category: values[categoryIndex],
            });
          }
        }
        
        if (tickets.length < 10) {
          throw new Error('Need at least 10 categorized tickets for training');
        }
        
        // Extract unique categories
        const categories = [...new Set(tickets.map(t => t.category))];
        
        // Train the classifier
        const result = await trainClassifier(tickets, categories);
        setTrainingResult(result);
        enqueueSnackbar('Classifier trained successfully!', { variant: 'success' });
      };
      reader.onerror = () => {
        throw new Error('Error reading the file');
      };
      reader.readAsText(file);
    } catch (error) {
      enqueueSnackbar(error.message, { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Classifier Training
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Training Data
              </Typography>
              
              <Typography variant="body2" paragraph>
                Upload a CSV file containing historical tickets with categories to train the classifier.
                The file should include at least subject, description, and category columns.
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
                    label="Category Column"
                    value={categoryCol}
                    onChange={(e) => setCategoryCol(e.target.value)}
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
                  
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={handleTrain}
                      disabled={loading || !file}
                      startIcon={loading ? <CircularProgress size={20} /> : null}
                    >
                      Train Classifier
                    </Button>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          {trainingResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Results
                </Typography>
                
                <Typography variant="body1" paragraph>
                  Successfully trained classifier on {trainingResult.message.split(' ')[4]} tickets.
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  Categories Learned
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {trainingResult.categories.map((category, index) => (
                    <Chip key={index} label={category} color="primary" variant="outlined" />
                  ))}
                </Box>
                
                <Box sx={{ mt: 3 }}>
                  <Alert severity="info">
                    The classifier will now be used for automatic categorization of new tickets.
                  </Alert>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default ClassifierTraining;