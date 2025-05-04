import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import SingleAnalysis from './components/SingleAnalysis';
import BulkAnalysis from './components/BulkAnalysis';
import TicketUpload from './components/TicketUpload';
import ClassifierTraining from './components/ClassifierTraining';
import ModelInfo from './components/ModelInfo';
import { SnackbarProvider } from 'notistack';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider maxSnack={3}>
        <Router>
          <div className="App">
            <Navbar />
            <main style={{ marginTop: '64px', padding: '24px' }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/single" element={<SingleAnalysis />} />
                <Route path="/bulk" element={<BulkAnalysis />} />
                <Route path="/upload" element={<TicketUpload />} />
                <Route path="/train" element={<ClassifierTraining />} />
                <Route path="/model" element={<ModelInfo />} />
              </Routes>
            </main>
          </div>
        </Router>
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default App;